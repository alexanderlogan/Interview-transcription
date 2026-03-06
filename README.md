# Interview Transcriber

A local, CPU-only interview transcription tool for Windows. Captures system
audio via WASAPI loopback, transcribes with Whisper, and diarizes with
pyannote to produce a named speaker transcript saved as structured JSON.

No cloud services. No subscription. Runs entirely on your machine.

## Features

- Single loopback stream captures all audio through your speakers
- Whisper `base` model transcription (CPU, int8, ~3s per 30s chunk)
- pyannote speaker diarization — separates speakers within each chunk
- Voice enrollment — identifies you by name in every session
- Named speaker labels — primary guest named at launch, extras auto-labelled
- Structured JSON session output with timestamps and confidence scores
- Session viewer — pretty-prints any session as a readable transcript

## Requirements

- Windows 10/11
- Python 3.10+
- WASAPI-compatible audio output device
- HuggingFace account with access to pyannote models (free)

## Installation

```powershell
git clone https://github.com/alexanderlogan/Interview-transcription.git
cd Interview-transcription
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your HuggingFace token as a permanent environment variable:

```powershell
[System.Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token", "User")
```

Accept model conditions at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/embedding

## Voice Enrollment (one time only)

Record 30-60 seconds of yourself speaking naturally (voice memo on phone works
well). Transfer the file to the project root, then run:

```powershell
python enroll.py --file your_voice.mp3 --token $env:HF_TOKEN
```

Your voice profile is saved to `profiles/host_embedding.npy` and reused
across all future sessions.

## Usage

Double-click `Interview Transcriber.bat` on your desktop, or from terminal:

```powershell
python main.py
```

At the prompt, type the name of your primary guest and press Enter.
The session begins immediately. Press Ctrl+C to stop.

## Viewing Transcripts

```powershell
python view_session.py                          # most recent session
python view_session.py sessions/session_X.json  # specific session
```

## Session Output

Sessions are saved to `sessions/session_YYYYMMDD_HHMMSS.json`:

```json
{
  "segments": [
    {
      "speaker": "Alex Logan",
      "text": "Tell me about your experience with...",
      "started_at": "2026-03-05T18:44:09Z",
      "duration_seconds": 3.1,
      "confidence": 0.74
    }
  ]
}
```

## Device Configuration

See [DEVICES.md](DEVICES.md) for instructions on configuring different
audio hardware.

## Project Structure

```
main.py                   Entry point
enroll.py                 One-time voice enrollment
view_session.py           Session transcript viewer
config.py                 Central configuration
src/
  audio/capture.py        WASAPI loopback capture
  diarization/diarizer.py pyannote diarization + speaker ID
  transcription/          Whisper engine
  pipeline.py             Diarization + transcription pipeline
  output/session.py       JSON session writer
sessions/                 Session output files
profiles/                 Voice embedding profiles
```

## Roadmap

- [ ] Real interview validation of host voice identification
- [ ] Whisper `small` model option for higher accuracy
- [ ] Export transcript as plain text or markdown
