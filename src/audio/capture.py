# src/audio/capture.py
#
# Dual WASAPI audio capture for Interview Transcriber.
#
# Stream A (Loopback):  Captures system audio output via callback-based stream.
#                       Callback pattern prevents blocking on silence.
#                       Targets Headphones (SK200) [Loopback] by name.
#                       Speaker label: SPEAKER_LOOPBACK ("Guest")
#
# Stream B (Microphone): Captures EMEET SmartCam Nova 4K microphone.
#                        Uses blocking read (mic always active, no silence issue).
#                        Speaker label: SPEAKER_MIC ("Host")
#
# Both streams:
#   - Source rate: read from device (SK200 loopback = 44100 Hz, EMEET = 48000 Hz)
#   - Output: 16kHz mono float32 numpy arrays (Whisper-ready)
#   - Resampling: soxr (high-quality, low-latency)
#   - Delivery: puts chunks into caller-supplied queue.Queue instances

import threading
import queue
import logging
import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from config import (
    WHISPER_SAMPLE_RATE,
    SOURCE_CHANNELS,
    MAX_QUEUE_SIZE,
    CHUNK_DURATION_SECONDS,
)

logger = logging.getLogger(__name__)

LOOPBACK_DEVICE_NAME = "SK200"
MIC_DEVICE_NAME = "EMEET SmartCam"


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def get_loopback_device(pa: pyaudio.PyAudio) -> dict:
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if (
            LOOPBACK_DEVICE_NAME.lower() in d["name"].lower()
            and "[Loopback]" in d["name"]
            and d["maxInputChannels"] > 0
        ):
            logger.info(
                "Loopback device selected: [%d] %s (%.0f Hz)",
                d["index"], d["name"], d["defaultSampleRate"],
            )
            return d
    raise RuntimeError(
        f"No loopback device found matching '{LOOPBACK_DEVICE_NAME}'. "
        "Ensure the SK200 is connected and set as the Windows default output device."
    )


def get_mic_device(pa: pyaudio.PyAudio) -> dict:
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if (
            MIC_DEVICE_NAME.lower() in d["name"].lower()
            and d["maxInputChannels"] > 0
        ):
            logger.info(
                "Microphone device selected: [%d] %s (%.0f Hz)",
                d["index"], d["name"], d["defaultSampleRate"],
            )
            return d
    raise RuntimeError(
        f"No input device found matching '{MIC_DEVICE_NAME}'. "
        "Ensure the EMEET SmartCam Nova 4K is connected."
    )


def enumerate_audio_devices(pa: pyaudio.PyAudio) -> None:
    count = pa.get_device_count()
    logger.debug("--- Audio Device Enumeration (%d devices) ---", count)
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        logger.debug(
            "  [%d] %s | in=%d out=%d | %.0f Hz",
            i, info["name"], info["maxInputChannels"],
            info["maxOutputChannels"], info["defaultSampleRate"],
        )
    logger.debug("--- End Device Enumeration ---")


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def raw_bytes_to_whisper_array(
    raw_bytes: bytes,
    source_rate: int,
    source_channels: int = SOURCE_CHANNELS,
    target_rate: int = WHISPER_SAMPLE_RATE,
) -> np.ndarray:
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    if source_channels > 1:
        audio_float32 = audio_float32.reshape(-1, source_channels).mean(axis=1)
    if source_rate != target_rate:
        audio_float32 = soxr.resample(audio_float32, source_rate, target_rate, quality="HQ")
    return audio_float32


# ---------------------------------------------------------------------------
# Capture threads
# ---------------------------------------------------------------------------

class LoopbackCaptureThread(threading.Thread):
    """
    Thread A: Captures WASAPI loopback from Headphones (SK200).

    Uses a callback-based PyAudio stream. The callback receives audio frames
    from WASAPI as they arrive, accumulates them into 5-second chunks, then
    puts each chunk into the output queue. This pattern never blocks on silence
    — WASAPI delivers silence frames when no audio is playing, keeping the
    thread alive and responsive to stop_event.
    """

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="LoopbackCapture", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._device_info: dict | None = None
        self._buffer = b""
        self._native_rate: int = 0
        self._frames_per_chunk: int = 0

    @property
    def device_name(self) -> str:
        return self._device_info["name"] if self._device_info else "unknown"

    def _callback(self, in_data, frame_count, time_info, status):
        """
        Called by PyAudio on each audio frame from WASAPI.
        Accumulates raw bytes until we have a full 5-second chunk,
        then enqueues for transcription.
        """
        self._buffer += in_data
        bytes_per_chunk = self._frames_per_chunk * SOURCE_CHANNELS * 2  # int16 = 2 bytes

        while len(self._buffer) >= bytes_per_chunk:
            chunk_bytes = self._buffer[:bytes_per_chunk]
            self._buffer = self._buffer[bytes_per_chunk:]
            audio = raw_bytes_to_whisper_array(
                chunk_bytes,
                source_rate=self._native_rate,
                source_channels=SOURCE_CHANNELS,
            )
            self._enqueue(audio)

        return (None, pyaudio.paContinue)

    def run(self) -> None:
        pa = pyaudio.PyAudio()
        try:
            enumerate_audio_devices(pa)
            self._device_info = get_loopback_device(pa)
            self._native_rate = int(self._device_info["defaultSampleRate"])
            self._frames_per_chunk = self._native_rate * CHUNK_DURATION_SECONDS

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=SOURCE_CHANNELS,
                rate=self._native_rate,
                input=True,
                input_device_index=self._device_info["index"],
                frames_per_buffer=1024,
                stream_callback=self._callback,
            )

            stream.start_stream()
            logger.info("Loopback capture started (%.0f Hz).", self._native_rate)

            while not self.stop_event.is_set() and stream.is_active():
                self.stop_event.wait(timeout=0.1)

            stream.stop_stream()
            stream.close()
            logger.info("Loopback capture stopped.")
        except Exception as exc:
            logger.error("Loopback capture error: %s", exc, exc_info=True)
        finally:
            pa.terminate()

    def _enqueue(self, audio: np.ndarray) -> None:
        if self.output_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                self.output_queue.get_nowait()
                logger.warning("Loopback queue full — dropped oldest chunk.")
            except queue.Empty:
                pass
        self.output_queue.put(audio)


class MicrophoneCaptureThread(threading.Thread):
    """
    Thread B: Captures EMEET SmartCam Nova 4K microphone.
    Uses blocking reads — microphone is always active, no silence issue.
    Puts float32 mono 16kHz numpy arrays into output_queue.
    """

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="MicrophoneCapture", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._device_info: dict | None = None

    @property
    def device_name(self) -> str:
        return self._device_info["name"] if self._device_info else "unknown"

    def run(self) -> None:
        pa = pyaudio.PyAudio()
        try:
            self._device_info = get_mic_device(pa)
            native_rate = int(self._device_info["defaultSampleRate"])
            frames_per_chunk = native_rate * CHUNK_DURATION_SECONDS

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=native_rate,
                input=True,
                input_device_index=self._device_info["index"],
                frames_per_buffer=frames_per_chunk,
            )

            logger.info("Microphone capture started (%.0f Hz).", native_rate)
            while not self.stop_event.is_set():
                try:
                    raw = stream.read(frames_per_chunk, exception_on_overflow=False)
                    audio = raw_bytes_to_whisper_array(
                        raw, source_rate=native_rate, source_channels=1,
                    )
                    self._enqueue(audio)
                except OSError as exc:
                    logger.error("Microphone read error: %s", exc)
                    break

            stream.stop_stream()
            stream.close()
            logger.info("Microphone capture stopped.")
        except Exception as exc:
            logger.error("Microphone capture error: %s", exc, exc_info=True)
        finally:
            pa.terminate()

    def _enqueue(self, audio: np.ndarray) -> None:
        if self.output_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                self.output_queue.get_nowait()
                logger.warning("Microphone queue full — dropped oldest chunk.")
            except queue.Empty:
                pass
        self.output_queue.put(audio)
