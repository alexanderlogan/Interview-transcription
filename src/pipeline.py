# src/pipeline.py
#
# Transcription pipeline for Interview Transcriber — Phase 3.
#
# Architecture:
#
#   LoopbackCaptureThread --> loopback_queue --> DiarizationTranscriptionThread
#                                                         |
#                                                    Diarizer (pyannote)
#                                                         |
#                                                 [speaker turns]
#                                                         |
#                                                  WhisperEngine
#                                                         |
#                                                   SessionWriter
#
# Each 30-second chunk is first diarized into speaker turns, then each
# turn is transcribed independently by Whisper. Results are written to
# the session JSON with resolved speaker names.

import threading
import queue
import logging
from datetime import datetime, timezone

from src.transcription.whisper_engine import WhisperEngine
from src.diarization.diarizer import Diarizer
from src.output.session import SessionWriter

logger = logging.getLogger(__name__)


class DiarizationTranscriptionThread(threading.Thread):
    """
    Consumes 30-second audio chunks from the loopback queue.
    For each chunk:
      1. Diarizes into speaker turns via pyannote
      2. Transcribes each turn via Whisper
      3. Writes non-empty results to session JSON
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        whisper: WhisperEngine,
        diarizer: Diarizer,
        session: SessionWriter,
        stop_event: threading.Event,
    ):
        super().__init__(name="DiarizationTranscription", daemon=True)
        self.audio_queue = audio_queue
        self.whisper = whisper
        self.diarizer = diarizer
        self.session = session
        self.stop_event = stop_event
        self._chunks_processed = 0
        self._segments_written = 0
        self._session_elapsed = 0.0  # Running total of audio time processed

    def run(self) -> None:
        logger.info("Diarization+Transcription thread started.")

        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._chunks_processed += 1
            chunk_start_time = self._session_elapsed
            chunk_duration = len(audio) / 16000
            self._session_elapsed += chunk_duration

            logger.info(
                "Processing chunk %d (%.0f-%.0fs of session)...",
                self._chunks_processed,
                chunk_start_time,
                self._session_elapsed,
            )

            # Step 1: Diarize into speaker turns
            try:
                turns = self.diarizer.diarize(audio, chunk_start_time=chunk_start_time)
            except Exception as exc:
                logger.error("Diarization error on chunk %d: %s", self._chunks_processed, exc, exc_info=True)
                continue

            if not turns:
                logger.info("No speaker turns detected in chunk %d.", self._chunks_processed)
                continue

            # Step 2: Transcribe each speaker turn independently
            for turn in turns:
                started_at = datetime.now(timezone.utc)

                try:
                    result = self.whisper.transcribe(turn.audio)
                except Exception as exc:
                    logger.error("Transcription error: %s", exc, exc_info=True)
                    continue

                if result.was_silent or not result.text:
                    continue

                self._segments_written += 1
                self.session.append_segment(
                    speaker=turn.speaker_label,
                    text=result.text,
                    confidence=result.confidence,
                    source_stream="loopback",
                    started_at=started_at,
                    duration_seconds=turn.end_seconds - turn.start_seconds,
                )

        logger.info(
            "Pipeline thread stopped. Processed %d chunks, wrote %d segments.",
            self._chunks_processed,
            self._segments_written,
        )
