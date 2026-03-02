# src/pipeline.py
#
# Transcription pipeline for Interview Transcriber.
#
# Phase 2 architecture (single loopback stream):
#
#   LoopbackCaptureThread  -->  loopback_queue  -->  TranscriptionThread
#                                                          |
#                                                    WhisperEngine
#                                                          |
#                                                    SessionWriter
#
# TranscriptionThread pulls audio chunks from the loopback queue,
# passes them to WhisperEngine, and writes results to the session JSON.
#
# Speaker label is placeholder "Speaker" in Phase 2.
# Phase 3 will replace this with pyannote diarization and named speakers.

import threading
import queue
import logging
from datetime import datetime, timezone

from src.transcription.whisper_engine import WhisperEngine
from src.output.session import SessionWriter
from config import SPEAKER_LOOPBACK

logger = logging.getLogger(__name__)

# Placeholder label until Phase 3 diarization is integrated
PHASE2_SPEAKER_LABEL = "Speaker"


class TranscriptionThread(threading.Thread):
    """
    Consumes audio chunks from the loopback queue, transcribes via Whisper,
    and writes non-empty results to the session JSON file.

    Runs until stop_event is set and the queue is drained.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        whisper: WhisperEngine,
        session: SessionWriter,
        stop_event: threading.Event,
    ):
        super().__init__(name="Transcription", daemon=True)
        self.audio_queue = audio_queue
        self.whisper = whisper
        self.session = session
        self.stop_event = stop_event
        self._chunks_processed = 0
        self._chunks_transcribed = 0

    def run(self) -> None:
        logger.info("Transcription thread started.")
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            started_at = datetime.now(timezone.utc)
            self._chunks_processed += 1

            try:
                result = self.whisper.transcribe(audio)
            except Exception as exc:
                logger.error("Transcription error: %s", exc, exc_info=True)
                continue

            # Skip silent chunks and chunks with no detected speech
            if result.was_silent or not result.text:
                continue

            self._chunks_transcribed += 1
            self.session.append_segment(
                speaker=PHASE2_SPEAKER_LABEL,
                text=result.text,
                confidence=result.confidence,
                source_stream="loopback",
                started_at=started_at,
                duration_seconds=result.duration_seconds,
            )

        logger.info(
            "Transcription thread stopped. Processed %d chunks, transcribed %d.",
            self._chunks_processed,
            self._chunks_transcribed,
        )
