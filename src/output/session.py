# src/output/session.py
#
# Session file management for Interview Transcriber.
#
# Owns the canonical JSON schema for a transcript session.
# Phase 1: Initialises the session file with metadata on start.
#          Writes a clean closure record on stop.
# Phase 2+: Transcript entries (segments) will be appended via append_segment().
#
# JSON Schema (top-level):
# {
#   "schema_version": "1.0",
#   "session_id": "20250220_143022",
#   "started_at": "2025-02-20T14:30:22.431Z",
#   "ended_at": "2025-02-20T14:55:10.112Z",   # null until session closes
#   "devices": {
#     "loopback": "Speakers (Realtek Audio)",
#     "microphone": "Microphone (USB SK200)"
#   },
#   "config": {
#     "whisper_model": "base",
#     "language": "en",
#     "chunk_duration_seconds": 5
#   },
#   "segments": [
#     {
#       "segment_id": 1,
#       "speaker": "Guest",
#       "started_at": "2025-02-20T14:30:27.000Z",
#       "duration_seconds": 4.96,
#       "text": "Tell me about yourself.",
#       "confidence": 0.94,
#       "source_stream": "loopback"
#     },
#     ...
#   ]
# }

import json
import os
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from config import (
    SESSIONS_DIR,
    JSON_INDENT,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    CHUNK_DURATION_SECONDS,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


class SessionWriter:
    """
    Manages the lifecycle of a single transcript session JSON file.
    Thread-safe: uses a lock for all write operations.
    """

    def __init__(self, loopback_device_name: str, mic_device_name: str):
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._session_id = self._started_at.strftime("%Y%m%d_%H%M%S")
        self._segment_counter = 0

        sessions_path = Path(SESSIONS_DIR)
        sessions_path.mkdir(parents=True, exist_ok=True)
        self._filepath = sessions_path / f"session_{self._session_id}.json"

        self._data = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self._session_id,
            "started_at": self._started_at.isoformat(),
            "ended_at": None,
            "devices": {
                "loopback": loopback_device_name,
                "microphone": mic_device_name,
            },
            "config": {
                "whisper_model": WHISPER_MODEL,
                "language": WHISPER_LANGUAGE,
                "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
            },
            "segments": [],
        }

        self._flush()
        logger.info("Session initialised: %s", self._filepath)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def filepath(self) -> Path:
        return self._filepath

    def append_segment(
        self,
        speaker: str,
        text: str,
        confidence: float,
        source_stream: str,
        started_at: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> int:
        """
        Append a transcribed segment to the session.
        Returns the assigned segment_id.
        Called by Phase 2 transcription pipeline.
        """
        with self._lock:
            self._segment_counter += 1
            segment = {
                "segment_id": self._segment_counter,
                "speaker": speaker,
                "started_at": (started_at or datetime.now(timezone.utc)).isoformat(),
                "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
                "text": text.strip(),
                "confidence": round(confidence, 4),
                "source_stream": source_stream,
            }
            self._data["segments"].append(segment)
            self._flush()
            return self._segment_counter

    def close(self) -> None:
        """
        Mark the session as ended and write the final state to disk.
        """
        with self._lock:
            self._data["ended_at"] = datetime.now(timezone.utc).isoformat()
            self._flush()
        logger.info(
            "Session closed: %s (%d segments)", self._filepath, self._segment_counter
        )

    def _flush(self) -> None:
        """Write current state to disk. Caller must hold self._lock."""
        tmp_path = self._filepath.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=JSON_INDENT, ensure_ascii=False)
        os.replace(tmp_path, self._filepath)
