# config.py
# Central configuration for Interview Transcriber.

# --- Audio Capture ---
# SOURCE_SAMPLE_RATE removed — device native rates read at runtime
# SK200 loopback = 44100 Hz, EMEET mic = 48000 Hz

# Channels from WASAPI loopback source (stereo)
SOURCE_CHANNELS = 2

# Target sample rate for Whisper and pyannote (required)
WHISPER_SAMPLE_RATE = 16000

# Chunk duration in seconds — 30s for reliable diarization
CHUNK_DURATION_SECONDS = 30

# --- Queue Management ---
MAX_QUEUE_SIZE = 4

# --- Speaker Labels ---
SPEAKER_LOOPBACK = "Guest"   # System audio — the interviewer
SPEAKER_MIC = "Host"         # Microphone — you

# --- Output ---
SESSIONS_DIR = "sessions"
JSON_INDENT = 2

# --- Voice Profiles ---
PROFILES_DIR = "profiles"
HOST_NAME = "Alex Logan"

# --- Whisper ---
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = "en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# --- Pyannote ---
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# --- HuggingFace ---
# Set via environment variable HF_TOKEN or passed at enrollment
HF_TOKEN_ENV = "HF_TOKEN"
