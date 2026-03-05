# src/diarization/diarizer.py
#
# Speaker diarization and identification for Interview Transcriber.
#
# Uses pyannote/speaker-diarization-3.1 DiarizeOutput API:
#   result.speaker_diarization  -> pyannote Annotation (itertracks)
#   result.speaker_embeddings   -> np.ndarray (n_speakers, 256)
#
# Speaker identification:
#   - Compares each speaker's embedding against saved host profile
#   - Highest cosine similarity above threshold = HOST_NAME
#   - All others = guest_name provided at session start

import logging
import numpy as np
import torch
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import (
    PYANNOTE_MODEL,
    WHISPER_SAMPLE_RATE,
    PROFILES_DIR,
    HOST_NAME,
)

logger = logging.getLogger(__name__)

HOST_SIMILARITY_THRESHOLD = 0.65


@dataclass
class SpeakerTurn:
    """A single resolved speaker turn."""
    speaker_label: str
    start_seconds: float
    end_seconds: float
    audio: np.ndarray  # float32 mono 16kHz


class Diarizer:
    """
    Wraps pyannote speaker-diarization-3.1.
    Uses DiarizeOutput.speaker_embeddings for host identification
    without a separate Inference model call.
    """

    def __init__(self, hf_token: str, guest_name: str):
        self._hf_token = hf_token
        self._guest_name = guest_name
        self._pipeline = None
        self._host_embedding: Optional[np.ndarray] = None
        self._guest_label_map: dict = {}  # pyannote label -> resolved guest name

    def load(self) -> None:
        warnings.filterwarnings("ignore")
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline...")
        self._pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            token=self._hf_token,
        )
        self._pipeline.to(torch.device("cpu"))

        profile_path = Path(PROFILES_DIR) / "host_embedding.npy"
        if profile_path.exists():
            self._host_embedding = np.load(str(profile_path))
            logger.info("Host voice profile loaded: %s", profile_path)
        else:
            logger.warning(
                "No host voice profile found. "
                "Run enroll.py to enable speaker identification."
            )

        logger.info("Diarizer ready.")

    def diarize(self, audio: np.ndarray, chunk_start_time: float = 0.0) -> list[SpeakerTurn]:
        if self._pipeline is None:
            raise RuntimeError("Diarizer.load() must be called before diarize().")

        if len(audio) == 0:
            return []

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        input_data = {"waveform": audio_tensor, "sample_rate": WHISPER_SAMPLE_RATE}

        try:
            output = self._pipeline(input_data)
        except Exception as exc:
            logger.error("Diarization failed: %s", exc, exc_info=True)
            return []

        # Extract annotation and embeddings from DiarizeOutput
        annotation = output.speaker_diarization
        speaker_embeddings = output.speaker_embeddings  # shape: (n_speakers, 256)

        # Collect raw turns from annotation
        raw_turns = []
        try:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                raw_turns.append((turn.start, turn.end, speaker))
        except Exception as exc:
            logger.error("Error reading diarization annotation: %s", exc, exc_info=True)
            return []

        if not raw_turns:
            logger.debug("No speaker turns detected in chunk.")
            return []

        # Get ordered list of unique speakers from annotation
        unique_speakers = annotation.labels()

        # Identify host using pre-computed embeddings from DiarizeOutput
        host_speaker_label = self._identify_host(unique_speakers, speaker_embeddings)

        # Build resolved SpeakerTurn objects
        logger.info("Raw turns from annotation: %d", len(raw_turns))
        logger.info("Unique speakers in chunk: %s", unique_speakers)
        logger.info("Host identified as: %s", host_speaker_label)

        # Pre-register all non-host speakers so labels are consistent within chunk
        for sp in unique_speakers:
            if sp != host_speaker_label:
                self._resolve_guest_name(sp)

        turns = []
        for start, end, pyannote_label in raw_turns:
            start_sample = int(start * WHISPER_SAMPLE_RATE)
            end_sample = int(end * WHISPER_SAMPLE_RATE)
            audio_slice = audio[start_sample:end_sample]

            if len(audio_slice) < WHISPER_SAMPLE_RATE:  # Skip turns under 1 second
                continue

            if host_speaker_label and pyannote_label == host_speaker_label:
                resolved_name = HOST_NAME
            else:
                resolved_name = self._resolve_guest_name(pyannote_label)

            turns.append(SpeakerTurn(
                speaker_label=resolved_name,
                start_seconds=chunk_start_time + start,
                end_seconds=chunk_start_time + end,
                audio=audio_slice,
            ))

        logger.info(
            "Diarized chunk: %d turns (%s)",
            len(turns),
            ", ".join(f"{t.speaker_label} {t.end_seconds - t.start_seconds:.1f}s" for t in turns),
        )
        return turns

    def _resolve_guest_name(self, pyannote_label: str) -> str:
        """
        Map a pyannote speaker label to a guest name.
        Primary guest uses the name provided at session start.
        Additional guests are labelled Guest 2, Guest 3, etc.
        """
        if pyannote_label not in self._guest_label_map:
            if not self._guest_label_map:
                # First unknown speaker = primary guest
                self._guest_label_map[pyannote_label] = self._guest_name
            else:
                # Additional speakers = Guest 2, Guest 3, etc.
                n = len(self._guest_label_map) + 1
                self._guest_label_map[pyannote_label] = f"Guest {n}"
                logger.info("Additional speaker detected — labelled '%s'", self._guest_label_map[pyannote_label])
        return self._guest_label_map[pyannote_label]

    def _identify_host(
        self,
        unique_speakers: list,
        speaker_embeddings: np.ndarray,
    ) -> Optional[str]:
        """
        Compare each speaker's embedding against saved host profile.
        Returns the pyannote speaker label with highest cosine similarity,
        or None if below threshold or no profile loaded.
        """
        if self._host_embedding is None:
            return None

        if speaker_embeddings is None or len(speaker_embeddings) == 0:
            return None

        best_label = None
        best_similarity = -1.0

        for i, speaker in enumerate(unique_speakers):
            if i >= len(speaker_embeddings):
                break
            embedding = speaker_embeddings[i]
            similarity = self._cosine_similarity(self._host_embedding, embedding)
            logger.debug("Speaker %s similarity to host: %.3f", speaker, similarity)

            if similarity > best_similarity:
                best_similarity = similarity
                best_label = speaker

        if best_similarity >= HOST_SIMILARITY_THRESHOLD:
            logger.info(
                "Host identified as '%s' (similarity=%.3f)", best_label, best_similarity
            )
            return best_label
        else:
            logger.info(
                "Host not identified — best similarity %.3f below threshold %.3f",
                best_similarity, HOST_SIMILARITY_THRESHOLD,
            )
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
