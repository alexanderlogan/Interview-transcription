# tests/test_audio_processing.py
#
# Unit tests for audio processing utilities that can run WITHOUT hardware.
# Tests the raw_bytes_to_whisper_array conversion pipeline.
# Hardware-dependent capture tests are covered by manual verification (main.py).

import numpy as np
import pytest

from src.audio.capture import raw_bytes_to_whisper_array
from config import SOURCE_SAMPLE_RATE, WHISPER_SAMPLE_RATE


def make_sine_int16(frequency: float, duration: float, rate: int, channels: int = 2) -> bytes:
    """Generate a sine wave as int16 PCM bytes for testing."""
    n_samples = int(rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    sine = np.sin(2 * np.pi * frequency * t)
    if channels > 1:
        # Interleave identical channels (stereo)
        sine = np.repeat(sine[:, np.newaxis], channels, axis=1).flatten()
    int16 = (sine * 32767).astype(np.int16)
    return int16.tobytes()


class TestRawBytesToWhisperArray:

    def test_output_dtype_is_float32(self):
        raw = make_sine_int16(440, 1.0, SOURCE_SAMPLE_RATE, channels=2)
        result = raw_bytes_to_whisper_array(raw)
        assert result.dtype == np.float32

    def test_output_is_mono(self):
        raw = make_sine_int16(440, 1.0, SOURCE_SAMPLE_RATE, channels=2)
        result = raw_bytes_to_whisper_array(raw)
        assert result.ndim == 1, "Output must be 1D (mono)"

    def test_output_sample_rate_is_16khz(self):
        duration = 1.0
        raw = make_sine_int16(440, duration, SOURCE_SAMPLE_RATE, channels=2)
        result = raw_bytes_to_whisper_array(raw)
        expected_samples = int(WHISPER_SAMPLE_RATE * duration)
        # Allow small rounding tolerance from soxr
        assert abs(len(result) - expected_samples) <= 10, (
            f"Expected ~{expected_samples} samples at 16kHz, got {len(result)}"
        )

    def test_output_amplitude_normalised(self):
        raw = make_sine_int16(440, 1.0, SOURCE_SAMPLE_RATE, channels=2)
        result = raw_bytes_to_whisper_array(raw)
        assert result.max() <= 1.0, "Amplitude must be normalised to [-1.0, 1.0]"
        assert result.min() >= -1.0

    def test_mono_source_passthrough(self):
        """Microphone capture provides mono; verify it processes correctly."""
        raw = make_sine_int16(440, 1.0, SOURCE_SAMPLE_RATE, channels=1)
        result = raw_bytes_to_whisper_array(raw, source_channels=1)
        assert result.ndim == 1
        assert result.dtype == np.float32

    def test_silence_produces_near_zero_output(self):
        raw = bytes(SOURCE_SAMPLE_RATE * 2 * 2)  # 1 second of silence, stereo int16
        result = raw_bytes_to_whisper_array(raw)
        assert np.allclose(result, 0.0, atol=1e-6)
