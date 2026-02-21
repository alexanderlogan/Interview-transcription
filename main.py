# main.py
#
# Interview Transcriber — Entry Point
#
# Phase 1 behaviour:
#   - Starts loopback capture thread (system audio / interviewer)
#   - Starts microphone capture thread (user)
#   - Initialises session JSON file with device metadata
#   - Drains both capture queues and logs chunk receipt to confirm audio flow
#   - Runs until Ctrl+C
#   - Closes session file cleanly on exit
#
# Phase 2 will replace the queue-drain loop with Whisper transcription threads.

import queue
import threading
import logging
import sys
import time

from src.audio.capture import LoopbackCaptureThread, MicrophoneCaptureThread
from src.output.session import SessionWriter
from config import WHISPER_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Capture verification drain loop
# ---------------------------------------------------------------------------

def drain_queue(
    capture_queue: queue.Queue,
    label: str,
    stop_event: threading.Event,
) -> None:
    """
    Consumer thread (Phase 1 only).
    Pulls audio chunks from the queue and logs receipt.
    Confirms that audio is flowing without invoking Whisper.
    Will be replaced by WhisperTranscriptionThread in Phase 2.
    """
    chunk_count = 0
    while not stop_event.is_set():
        try:
            audio = capture_queue.get(timeout=1.0)
            chunk_count += 1
            duration = len(audio) / WHISPER_SAMPLE_RATE
            logger.info(
                "[%s] Chunk %d received — %.2f s of audio (%.0f samples)",
                label,
                chunk_count,
                duration,
                len(audio),
            )
        except queue.Empty:
            continue
    logger.info("[%s] Drain thread exiting after %d chunks.", label, chunk_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Interview Transcriber — Phase 1 Capture Verification")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop.")

    stop_event = threading.Event()

    # Queues
    loopback_queue: queue.Queue = queue.Queue()
    mic_queue: queue.Queue = queue.Queue()

    # Capture threads
    loopback_thread = LoopbackCaptureThread(loopback_queue, stop_event)
    mic_thread = MicrophoneCaptureThread(mic_queue, stop_event)

    # Start capture
    loopback_thread.start()
    mic_thread.start()

    # Brief pause to allow device detection to complete before reading device names
    time.sleep(1.5)

    # Initialise session file
    session = SessionWriter(
        loopback_device_name=loopback_thread.device_name,
        mic_device_name=mic_thread.device_name,
    )
    logger.info("Session file: %s", session.filepath)

    # Phase 1 drain threads (replaced by Whisper in Phase 2)
    loopback_drain = threading.Thread(
        target=drain_queue,
        args=(loopback_queue, "Loopback", stop_event),
        name="LoopbackDrain",
        daemon=True,
    )
    mic_drain = threading.Thread(
        target=drain_queue,
        args=(mic_queue, "Microphone", stop_event),
        name="MicDrain",
        daemon=True,
    )
    loopback_drain.start()
    mic_drain.start()

    # Run until Ctrl+C
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")

    # Graceful shutdown
    stop_event.set()
    loopback_thread.join(timeout=5)
    mic_thread.join(timeout=5)
    session.close()

    logger.info("Session saved: %s", session.filepath)
    logger.info("Done.")


if __name__ == "__main__":
    main()
