# src/audio/capture.py
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


def get_loopback_device(pa):
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if (
            LOOPBACK_DEVICE_NAME.lower() in d["name"].lower()
            and "[Loopback]" in d["name"]
            and d["maxInputChannels"] > 0
        ):
            logger.info("Loopback device selected: [%d] %s (%.0f Hz)", d["index"], d["name"], d["defaultSampleRate"])
            return d
    raise RuntimeError(f"No loopback device found matching '{LOOPBACK_DEVICE_NAME}'. Ensure the SK200 is connected.")


def get_mic_device(pa):
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if (
            MIC_DEVICE_NAME.lower() in d["name"].lower()
            and d["maxInputChannels"] > 0
        ):
            logger.info("Microphone device selected: [%d] %s (%.0f Hz)", d["index"], d["name"], d["defaultSampleRate"])
            return d
    raise RuntimeError(f"No input device found matching '{MIC_DEVICE_NAME}'. Ensure the EMEET SmartCam Nova 4K is connected.")


def enumerate_audio_devices(pa):
    count = pa.get_device_count()
    logger.debug("--- Audio Device Enumeration (%d devices) ---", count)
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        logger.debug("  [%d] %s | in=%d out=%d | %.0f Hz", i, info["name"], info["maxInputChannels"], info["maxOutputChannels"], info["defaultSampleRate"])
    logger.debug("--- End Device Enumeration ---")


def raw_bytes_to_whisper_array(raw_bytes, source_rate, source_channels=SOURCE_CHANNELS, target_rate=WHISPER_SAMPLE_RATE):
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    if source_channels > 1:
        audio_float32 = audio_float32.reshape(-1, source_channels).mean(axis=1)
    if source_rate != target_rate:
        audio_float32 = soxr.resample(audio_float32, source_rate, target_rate, quality="HQ")
    return audio_float32


class LoopbackCaptureThread(threading.Thread):
    def __init__(self, output_queue, stop_event):
        super().__init__(name="LoopbackCapture", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._device_info = None

    @property
    def device_name(self):
        return self._device_info["name"] if self._device_info else "unknown"

    def run(self):
        pa = pyaudio.PyAudio()
        try:
            enumerate_audio_devices(pa)
            self._device_info = get_loopback_device(pa)
            native_rate = int(self._device_info["defaultSampleRate"])
            frames_per_chunk = native_rate * CHUNK_DURATION_SECONDS
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=SOURCE_CHANNELS,
                rate=native_rate,
                input=True,
                input_device_index=self._device_info["index"],
                frames_per_buffer=frames_per_chunk,
            )
            logger.info("Loopback capture started (%.0f Hz).", native_rate)
            while not self.stop_event.is_set():
                try:
                    raw = stream.read(frames_per_chunk, exception_on_overflow=False)
                    audio = raw_bytes_to_whisper_array(raw, source_rate=native_rate, source_channels=SOURCE_CHANNELS)
                    self._enqueue(audio)
                except OSError as exc:
                    logger.error("Loopback read error: %s", exc)
                    break
            stream.stop_stream()
            stream.close()
            logger.info("Loopback capture stopped.")
        finally:
            pa.terminate()

    def _enqueue(self, audio):
        if self.output_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                self.output_queue.get_nowait()
                logger.warning("Loopback queue full — dropped oldest chunk.")
            except queue.Empty:
                pass
        self.output_queue.put(audio)


class MicrophoneCaptureThread(threading.Thread):
    def __init__(self, output_queue, stop_event):
        super().__init__(name="MicrophoneCapture", daemon=True)
        self.output_queue = output_queue
        self.stop_event = stop_event
        self._device_info = None

    @property
    def device_name(self):
        return self._device_info["name"] if self._device_info else "unknown"

    def run(self):
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
                    audio = raw_bytes_to_whisper_array(raw, source_rate=native_rate, source_channels=1)
                    self._enqueue(audio)
                except OSError as exc:
                    logger.error("Microphone read error: %s", exc)
                    break
            stream.stop_stream()
            stream.close()
            logger.info("Microphone capture stopped.")
        finally:
            pa.terminate()

    def _enqueue(self, audio):
        if self.output_queue.qsize() >= MAX_QUEUE_SIZE:
            try:
                self.output_queue.get_nowait()
                logger.warning("Microphone queue full — dropped oldest chunk.")
            except queue.Empty:
                pass
        self.output_queue.put(audio)
