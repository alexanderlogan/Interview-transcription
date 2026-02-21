# src/audio/capture.py
#
# Dual WASAPI audio capture for Interview Transcriber.
#
# Stream A (Loopback):  Captures system audio output (what speakers play).
#                       Uses PyAudioWPatch WASAPI loopback on default output device.
#                       Speaker label: SPEAKER_LOOPBACK ("Guest")
#
# Stream B (Microphone): Captures default Windows input device.
#                        Uses standard PyAudio input stream.
#                        Speaker label: SPEAKER_MIC ("Host")
#
# Both streams:
#   - Source: 48kHz stereo (WASAPI default)
#   - Output: 16kHz mono float32 numpy arrays (Whisper-ready)
#   - Resampling: soxr (high-quality, low-latency)
#   - Delivery: puts chunks into caller-supplied queue.Queue instances
#
# Carried over and extended from interpreter-verify-ru/src/audio/capture.py

import threading
import queue
import logging
import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from config import (
    SOURCE_SAMPLE_RATE,
    WHISPER_SAMPLE_RATE,
    SOURCE_CHANNELS,
    FRAMES_PER_CHUNK,
    MAX_QUEUE_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device discovery helpers
# ---------------------------------------------------------------------------

def get_default_loopback_device(pa: pyaudio.PyAudio) -> dict:
    """
    Return the WASAPI loopback device info for the current default output device.
    Raises RuntimeError if no loopback device is found.
    """
    try:
        default_output = pa.get_default_wasapi_loopback()
        logger.info(
            "Loopback device selected: [%d] %s (%.0f Hz)",
            default_output["index"],
            default_output["name"],
            default_output["defaultSampleRate"],
        )
        return default_output
    except OSError as exc:
        raise RuntimeError(
            "No WASAPI loopback device found. "
            "Ensure PyAudioWPatch is installed and a default output device is active."
        ) from exc


def get_default_input_device(pa: pyaudio.PyAudio) -> dict:
    """
    Return device info for the default Windows input device (microphone).
    Raises RuntimeError if no input device is found.
    """
    try:
        index = pa.get_default_input_device_info()["index"]
        info = pa.get_device_info_by_index(index)
        logger.info(
            "Microphone device selected: [%d] %s (%.0f Hz)",
            info["index"],
            info["name"],
            info["defaultSampleRate"],
        )
        return info
    except OSError as exc:
        raise RuntimeError(
            "No default input device found. "
            "Ensure a microphone is connected and set as the Windows default input device."
        ) from exc


def enumerate_audio_devices(pa: pyaudio.PyAudio) -> None:
    """
    Log all available audio devices. Useful for diagnostics.
    Call this at startup when DEBUG logging is enabled.
    """
    count = pa.get_device_count()
    logger.debug("--- Audio Device Enumeration (%d devices) ---", count)
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        logger.debug(
            "  [%d] %s | in=%d out=%d | %.0f Hz",
            i,
            info["name"],
            info["maxInputChannels"],
            info["maxOutputChannels"],
            info["defaultSampleRate"],
        )
    logger.debug("--- End Device Enumeration ---")


# ---------------------------------------------------------------------------
# Audio processing utility
# ---------------------------------------------------------------------------

def raw_bytes_to_whisper_array(
    raw_bytes: bytes,
    source_rate: int = SOURCE_SAMPLE_RATE,
    source_channels: int = SOURCE_CHANNELS,
    target_rate: int = WHISPER_SAMPLE_RATE,
) -> np.ndarray:
    """
    Convert raw PCM bytes (int16, stereo, source_rate) to a float32 mono
    numpy array at target_rate, ready for Whisper ingestion.

    Steps:
      1. Decode int16 LE bytes to numpy int16
      2. Convert to float32 in [-1.0, 1.0]
      3. Downmix stereo to mono by averaging channels
      4. Resample from source_rate to target_rate via soxr
    """
    # Step 1: decode
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

    # Step 2: normalise to float32
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # Step 3: stereo to mono
    if source_channels > 1:
        audio_float32 = audio_float32.reshape(-1, source_channels).mean(axis=1)

    # Step 4: resample
    if source_rate != target_rate:
        audio_float32 = soxr.resample(
            audio_float32, source_rate, target_rate, quality="HQ"
        )

    return audio_float32


# ---------------------------------------------------------------------------
# Capture threads
# ---------------------------------------------------------------------------

class LoopbackCaptureThread(threading.Thread):
    """
    Thread A: Captures WASAPI loopback (system audio output).
    Puts float32 mono 16kHz numpy arrays into output_queue.
    Runs until stop_event is set.
    """

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="LoopbackCapture", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._device_info: dict | None = None

    @property
    def device_name(self) -> str:
        return self._device_info["name"] if self._device_info else "unknown"

    def run(self) -> None:
        pa = pyaudio.PyAudio()
        try:
            enumerate_audio_devices(pa)
            self._device_info = get_default_loopback_device(pa)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=SOURCE_CHANNELS,
                rate=SOURCE_SAMPLE_RATE,
                input=True,
                input_device_index=self._device_info["index"],
                frames_per_buffer=FRAMES_PER_CHUNK,
            )

            logger.info("Loopback capture started.")
            while not self.stop_event.is_set():
                try:
                    raw = stream.read(FRAMES_PER_CHUNK, exception_on_overflow=False)
                    audio = raw_bytes_to_whisper_array(raw)
                    self._enqueue(audio)
                except OSError as exc:
                    logger.error("Loopback read error: %s", exc)
                    break

            stream.stop_stream()
            stream.close()
            logger.info("Loopback capture stopped.")
        finally:
            pa.terminate()

    def _enqueue(self, audio: np.ndarray) -> None:
        if self.output_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                dropped = self.output_queue.get_nowait()
                logger.warning(
                    "Loopback queue full — dropped oldest chunk (%.1f s)",
                    len(dropped) / WHISPER_SAMPLE_RATE,
                )
            except queue.Empty:
                pass
        self.output_queue.put(audio)


class MicrophoneCaptureThread(threading.Thread):
    """
    Thread B: Captures default Windows input device (microphone).
    Puts float32 mono 16kHz numpy arrays into output_queue.
    Runs until stop_event is set.
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
            self._device_info = get_default_input_device(pa)

            # Microphone may report a different native sample rate; honour it.
            native_rate = int(self._device_info.get("defaultSampleRate", SOURCE_SAMPLE_RATE))

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,               # Request mono from mic directly where possible
                rate=native_rate,
                input=True,
                input_device_index=self._device_info["index"],
                frames_per_buffer=native_rate * 5,  # 5-second chunks, matched to loopback
            )

            logger.info("Microphone capture started (native rate: %d Hz).", native_rate)
            while not self.stop_event.is_set():
                try:
                    raw = stream.read(native_rate * 5, exception_on_overflow=False)
                    # Mic captured as mono int16; pass source_channels=1
                    audio = raw_bytes_to_whisper_array(
                        raw,
                        source_rate=native_rate,
                        source_channels=1,
                    )
                    self._enqueue(audio)
                except OSError as exc:
                    logger.error("Microphone read error: %s", exc)
                    break

            stream.stop_stream()
            stream.close()
            logger.info("Microphone capture stopped.")
        finally:
            pa.terminate()

    def _enqueue(self, audio: np.ndarray) -> None:
        if self.output_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                dropped = self.output_queue.get_nowait()
                logger.warning(
                    "Microphone queue full — dropped oldest chunk (%.1f s)",
                    len(dropped) / WHISPER_SAMPLE_RATE,
                )
            except queue.Empty:
                pass
        self.output_queue.put(audio)
