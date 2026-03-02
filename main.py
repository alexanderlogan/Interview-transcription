# main.py
#
# Interview Transcriber — Entry Point
#
# Phase 2 behaviour:
#   - Single loopback stream captures all audio (interviewer + user)
#   - Whisper base model transcribes each 5-second chunk
#   - Silent chunks are discarded before reaching Whisper
#   - Results written to structured JSON session file in real time
#   - Speaker label is placeholder "Speaker" until Phase 3 diarization
#   - Runs until Ctrl+C, closes session cleanly on exit

import queue
import logging
import sys
import time

from src.audio.capture import LoopbackCaptureThread
from src.transcription.whisper_engine import WhisperEngine
from src.pipeline import TranscriptionThread
from src.output.session import SessionWriter
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Interview Transcriber — Phase 2 Transcription")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop.")

    stop_event = threading.Event()
    loopback_queue: queue.Queue = queue.Queue()

    # Load Whisper model before starting capture
    whisper = WhisperEngine()
    whisper.load()

    # Start loopback capture
    loopback_thread = LoopbackCaptureThread(loopback_queue, stop_event)
    loopback_thread.start()

    # Allow device detection to complete before reading device name
    time.sleep(2.0)

    # Initialise session file
    session = SessionWriter(
        loopback_device_name=loopback_thread.device_name,
        mic_device_name="n/a",
    )
    logger.info("Session file: %s", session.filepath)

    # Start transcription thread
    transcription_thread = TranscriptionThread(
        audio_queue=loopback_queue,
        whisper=whisper,
        session=session,
        stop_event=stop_event,
    )
    transcription_thread.start()

    # Run until Ctrl+C
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")

    # Graceful shutdown — signal stop, wait for queue to drain
    stop_event.set()
    loopback_thread.join(timeout=5)
    transcription_thread.join(timeout=30)  # Allow time to drain remaining chunks
    session.close()

    logger.info("Session saved: %s", session.filepath)
    logger.info("Done.")


if __name__ == "__main__":
    main()
