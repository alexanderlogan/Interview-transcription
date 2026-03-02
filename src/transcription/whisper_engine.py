# src/transcription/whisper_engine.py
#
# Whisper transcription engine for Interview Transcriber.
#
# Wraps faster-whisper with CTranslate2 backend.
# Confirmed on target hardware: base model = 2.9s per 5s chunk (acceptable).
#
# Responsibilities:
#   - Load and hold the Whisper model (loaded once at startup)
#   - Accept float32 mono 16kHz numpy arrays
#   - Apply RMS energy threshold to skip silent chunks
#   - Return structured TranscriptionResult for each chunk
#
# Phase 3 will add pyannote speaker diarization upstream of this engine.

import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# RMS energy threshold below which a chunk is considered silent and skipped.
# Range: 0.0 to 1.0. Tune if legitimate quiet speech is being dropped.
SILENCE_THRESHOLD = 0.01


@dataclass
class TranscriptionResult:
    """Structured output from a single Whisper transcription."""
    text: str
    confidence: float          # Mean log-probability converted to 0-1 range
    duration_seconds: float    # Duration of the audio chunk
    processing_seconds: float  # Wall time taken to transcribe
    was_silent: bool           # True if chunk was skipped due to silence


class WhisperEngine:
    """
    Loads the Whisper model once and transcribes audio chunks on demand.
    Thread-safe for single-threaded use (one transcription at a time).
    For Phase 2, one WhisperEngine instance is shared by the pipeline thread.
    """

    def __init__(self):
        self._model: Optional[WhisperModel] = None

    def load(self) -> None:
        """
        Load the Whisper model. Call once at startup before transcribing.
        Downloads ~140 MB on first run, then uses local cache.
        """
        logger.info(
            "Loading Whisper model '%s' (%s, %s)...",
            WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        )
        t0 = time.time()
        self._model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        elapsed = time.time() - t0
        logger.info("Whisper model loaded in %.1f seconds.", elapsed)

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a float32 mono 16kHz numpy array.

        Returns a TranscriptionResult. If the chunk is silent (RMS below
        SILENCE_THRESHOLD), returns immediately with was_silent=True and
        empty text — Whisper is not invoked.

        Args:
            audio: float32 mono array at WHISPER_SAMPLE_RATE (16kHz)

        Returns:
            TranscriptionResult with text, confidence, timing metadata
        """
        if self._model is None:
            raise RuntimeError("WhisperEngine.load() must be called before transcribe().")

        duration = len(audio) / WHISPER_SAMPLE_RATE

        # RMS energy check — skip silence without invoking Whisper
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < SILENCE_THRESHOLD:
            logger.debug("Chunk skipped — RMS %.4f below threshold %.4f.", rms, SILENCE_THRESHOLD)
            return TranscriptionResult(
                text="",
                confidence=0.0,
                duration_seconds=duration,
                processing_seconds=0.0,
                was_silent=True,
            )

        t0 = time.time()
        segments, info = self._model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            beam_size=5,
            vad_filter=True,           # Whisper built-in VAD as secondary filter
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Collect segments and compute mean confidence
        texts = []
        log_probs = []
        for segment in segments:
            if segment.text.strip():
                texts.append(segment.text.strip())
            if segment.avg_logprob is not None:
                log_probs.append(segment.avg_logprob)

        text = " ".join(texts)
        processing_seconds = time.time() - t0

        # Convert mean log-probability to a 0-1 confidence score
        # avg_logprob is typically in range [-1.0, 0.0]; clip and normalise
        if log_probs:
            mean_logprob = np.mean(log_probs)
            confidence = float(np.clip(1.0 + mean_logprob, 0.0, 1.0))
        else:
            confidence = 0.0

        if text:
            logger.info(
                "Transcribed %.1fs audio in %.1fs — \"%s\" (conf=%.2f)",
                duration, processing_seconds, text[:60], confidence,
            )
        else:
            logger.debug(
                "Transcribed %.1fs audio in %.1fs — no speech detected.",
                duration, processing_seconds,
            )

        return TranscriptionResult(
            text=text,
            confidence=confidence,
            duration_seconds=duration,
            processing_seconds=processing_seconds,
            was_silent=False,
        )
