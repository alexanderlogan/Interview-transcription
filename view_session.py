# view_session.py
#
# Session transcript viewer for Interview Transcriber.
#
# Reads a session JSON file and prints a clean, formatted transcript
# with speaker names, timestamps, and flowing text.
#
# Usage:
#   python view_session.py                        # most recent session
#   python view_session.py sessions/session_X.json  # specific session

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone


def format_timestamp(iso_string: str) -> str:
    """Convert ISO timestamp to HH:MM:SS format."""
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return "??:??:??"


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def find_latest_session() -> Path:
    """Find the most recently modified session file."""
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        raise FileNotFoundError("No sessions directory found.")
    files = sorted(sessions_dir.glob("session_*.json"), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError("No session files found in sessions/")
    return files[0]


def load_session(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_transcript(session: dict, path: Path) -> None:
    segments = session.get("segments", [])
    started_at = session.get("started_at", "")
    ended_at = session.get("ended_at", "")
    guest_name = session.get("guest_name", "Guest")
    devices = session.get("devices", {})
    config = session.get("config", {})

    # Calculate session duration
    try:
        start_dt = datetime.fromisoformat(started_at)
        end_dt = datetime.fromisoformat(ended_at)
        duration_secs = (end_dt - start_dt).total_seconds()
        duration_str = format_duration(duration_secs)
    except Exception:
        duration_str = "unknown"

    # Header
    print()
    print("=" * 70)
    print("  INTERVIEW TRANSCRIPT")
    print("=" * 70)
    print(f"  Session:   {session.get('session_id', 'unknown')}")
    print(f"  Date:      {format_timestamp(started_at)[:8]} — {started_at[:10]}")
    print(f"  Duration:  {duration_str}")
    print(f"  Speakers:  Alex Logan  |  {guest_name}")
    print(f"  Device:    {devices.get('loopback', 'unknown')}")
    print(f"  Model:     Whisper {config.get('whisper_model', 'unknown')}")
    print(f"  File:      {path.name}")
    print("=" * 70)
    print()

    if not segments:
        print("  No segments recorded in this session.")
        print()
        return

    # Print segments grouped by speaker turn
    prev_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        started = seg.get("started_at", "")
        duration = seg.get("duration_seconds", 0)
        confidence = seg.get("confidence", 0)

        if not text:
            continue

        # Print speaker header on speaker change
        if speaker != prev_speaker:
            if prev_speaker is not None:
                print()
            timestamp = format_timestamp(started)
            print(f"  [{timestamp}]  {speaker.upper()}")
            print(f"  {'-' * 50}")
            prev_speaker = speaker

        # Print text with confidence indicator
        conf_indicator = "●" if confidence >= 0.6 else "○"
        print(f"  {conf_indicator}  {text}")

    print()
    print("=" * 70)
    print(f"  {len(segments)} segments  |  Confidence: ● ≥0.6   ○ <0.6")
    print("=" * 70)
    print()


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
    else:
        try:
            path = find_latest_session()
            print(f"Loading most recent session: {path.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    try:
        session = load_session(path)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in session file: {e}")
        sys.exit(1)

    print_transcript(session, path)


if __name__ == "__main__":
    main()
