# main.py
#
# Interview Transcriber — Entry Point — Phase 3
#
# Launch sequence:
#   1. Check for saved host voice profile
#   2. Prompt for guest name
#   3. Load Whisper model
#   4. Load pyannote diarization pipeline
#   5. Start loopback capture (30-second chunks)
#   6. Run diarization + transcription pipeline
#   7. Write named speaker segments to session JSON
#   8. Ctrl+C for graceful shutdown

import os
import queue
import logging
import sys
import time
import threading
from pathlib import Path

from src.audio.capture import LoopbackCaptureThread
from src.transcription.whisper_engine import WhisperEngine
from src.diarization.diarizer import Diarizer
from src.pipeline import DiarizationTranscriptionThread
from src.output.session import SessionWriter
from config import PROFILES_DIR, HOST_NAME, HF_TOKEN_ENV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_hf_token() -> str:
    """Read HuggingFace token from environment variable."""
    token = os.environ.get(HF_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"HuggingFace token not found. "
            f"Set the {HF_TOKEN_ENV} environment variable:\n"
            f"  $env:{HF_TOKEN_ENV} = 'hf_...'"
        )
    return token


def check_voice_profile() -> bool:
    """Return True if a host voice profile exists."""
    profile_path = Path(PROFILES_DIR) / "host_embedding.npy"
    return profile_path.exists()


def prompt_guest_name() -> str:
    """Prompt for the name of today's interview guest."""
    print()
    print("=" * 60)
    name = input("Who are you meeting with today? ").strip()
    if not name:
        name = "Guest"
        logger.warning("No name provided — using default label 'Guest'.")
    print("=" * 60)
    print()
    return name


def main() -> None:
    logger.info("=" * 60)
    logger.info("Interview Transcriber — Phase 3 Diarization")
    logger.info("=" * 60)

    # Check voice profile
    if check_voice_profile():
        logger.info("Host voice profile found — speaker identification enabled.")
    else:
        logger.warning(
            "No host voice profile found. "
            "Run: python enroll.py --file YOUR_VOICE.m4a --token YOUR_HF_TOKEN"
        )
        logger.warning("Continuing without host identification — all speech labelled by turn only.")

    # Get guest name
    guest_name = prompt_guest_name()
    logger.info("Session participants: %s (host) + %s (guest)", HOST_NAME, guest_name)

    # Get HuggingFace token
    try:
        hf_token = get_hf_token()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    stop_event = threading.Event()
    loopback_queue: queue.Queue = queue.Queue()

    # Load models
    whisper = WhisperEngine()
    whisper.load()

    diarizer = Diarizer(hf_token=hf_token, guest_name=guest_name)
    diarizer.load()

    # Start loopback capture
    loopback_thread = LoopbackCaptureThread(loopback_queue, stop_event)
    loopback_thread.start()
    time.sleep(2.0)

    # Initialise session
    session = SessionWriter(
        loopback_device_name=loopback_thread.device_name,
        mic_device_name="n/a",
    )
    # Store guest name in session metadata
    session._data["guest_name"] = guest_name
    session._flush()

    logger.info("Session file: %s", session.filepath)
    logger.info("Recording started. Press Ctrl+C to stop.")

    # Start pipeline
    pipeline_thread = DiarizationTranscriptionThread(
        audio_queue=loopback_queue,
        whisper=whisper,
        diarizer=diarizer,
        session=session,
        stop_event=stop_event,
    )
    pipeline_thread.start()

    # Run until Ctrl+C
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")

    # Graceful shutdown
    stop_event.set()
    loopback_thread.join(timeout=5)
    pipeline_thread.join(timeout=60)  # Allow time to process remaining chunks
    session.close()

    logger.info("Session saved: %s", session.filepath)
    logger.info("Done.")


if __name__ == "__main__":
    main()
