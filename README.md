# Interview Transcriber

Real-time, two-stream interview transcription for personal learning and review.
Captures both system audio (interviewer) and microphone (you), produces a structured
JSON transcript with timestamps, speaker labels, and Whisper confidence scores.

## Hardware Requirements

- CPU-only inference (no GPU required)
- 16 GB RAM recommended
- Windows 11
- WASAPI-compatible audio devices

## Architecture

```
Stream A: WASAPI Loopback  ──► Queue A ──►  Whisper (base) ──► JSON Session File
Stream B: WASAPI Mic Input ──► Queue B ──►  Whisper (base) ──►
```

Two independent capture threads feed two independent transcription threads.
A single session writer thread merges and sorts output by timestamp.

## Phases

| Phase | Status      | Description                                 |
|-------|-------------|---------------------------------------------|
| 1     | In Progress | Repo scaffold + dual audio capture          |
| 2     | Planned     | Whisper transcription integration           |
| 3     | Planned     | Diarization benchmark + speaker label decision |
| 4     | Planned     | Polish, session management, v1.0 release    |

## Setup

```powershell
# One-time execution policy (if not already set)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip first (required for v26+ compatibility)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Usage (Phase 1 — Capture Verification)

```powershell
python main.py
```

Press `Ctrl+C` to stop. A session JSON file will be written to `sessions/`.

## Legal Notice

This tool is for personal use only. The user is responsible for obtaining
verbal consent from all parties prior to recording. Designed for use in
New Hampshire (two-party consent state) and other jurisdictions with
equivalent consent requirements. This tool does not inject audio or text
into any meeting platform and is invisible to other participants.

## Dependencies

See `requirements.txt`. Key packages:
- `PyAudioWPatch` — WASAPI loopback and input capture
- `faster-whisper` — CPU-optimized Whisper transcription (Phase 2+)
- `numpy`, `soxr` — Audio processing and resampling
