# enroll.py
#
# One-time voice enrollment for Interview Transcriber.
#
# Extracts your voice embedding from a provided audio file and saves it
# to profiles/host_embedding.npy. Run once before your first interview session.
#
# Uses pyannote/speaker-diarization-3.1 pipeline to extract embeddings
# in the same 256-dimension space used during diarization — ensuring
# direct comparability during speaker identification.
#
# Usage:
#   python enroll.py --file your_voice.m4a
#
# Audio file requirements:
#   - Format: any common format (m4a, mp3, wav, ogg)
#   - Duration: 30-60 seconds recommended
#   - Content: you speaking naturally, minimal background noise
#   - Source: voice memo from phone works perfectly

import argparse
import logging
import sys
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_audio_file(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio file as float32 mono numpy array."""
    import soundfile as sf

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        logger.info("Loaded audio: %.1f seconds at %d Hz", len(audio) / sr, sr)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")


def resample_to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample audio to 16kHz if needed."""
    if source_rate == 16000:
        return audio
    import soxr
    logger.info("Resampling from %d Hz to 16000 Hz...", source_rate)
    return soxr.resample(audio, source_rate, 16000, quality="HQ")


def enroll(audio_file: str, hf_token: str) -> None:
    from pyannote.audio import Pipeline

    profiles_dir = Path("profiles")
    profiles_dir.mkdir(exist_ok=True)
    output_path = profiles_dir / "host_embedding.npy"

    if output_path.exists():
        logger.warning("Existing profile found at %s.", output_path)
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != "y":
            logger.info("Enrollment cancelled.")
            return

    logger.info("Loading voice file: %s", audio_file)
    audio, sr = load_audio_file(audio_file)
    audio = resample_to_16k(audio, sr)

    if len(audio) / 16000 < 10:
        logger.warning("Audio is under 10 seconds. Recommend 30-60 seconds for reliable enrollment.")

    # Use the diarization pipeline to extract embeddings in the same
    # 256-dimension space used during session diarization
    logger.info("Loading pyannote diarization pipeline for enrollment...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device("cpu"))

    logger.info("Extracting voice embedding...")
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline({
        "waveform": audio_tensor,
        "sample_rate": 16000,
    })

    if output.speaker_embeddings is None or len(output.speaker_embeddings) == 0:
        raise RuntimeError(
            "No speaker detected in audio. "
            "Ensure the recording contains clear speech with no long silences."
        )

    # Use mean of all embeddings — enrollment audio should contain only your voice
    embedding_np = output.speaker_embeddings.mean(axis=0)
    logger.info(
        "Embedding extracted — shape: %s (256-dim, diarization space)",
        embedding_np.shape,
    )

    np.save(str(output_path), embedding_np)
    logger.info("Voice profile saved: %s", output_path)
    logger.info("Enrollment complete. You are ready to run main.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll host voice profile")
    parser.add_argument("--file", required=True, help="Path to voice recording (m4a, mp3, wav)")
    parser.add_argument("--token", required=True, help="HuggingFace access token")
    args = parser.parse_args()
    enroll(args.file, args.token)

