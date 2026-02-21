@'
# config.py
# Central configuration for Interview Transcriber.

# --- Audio Capture ---
# SOURCE_SAMPLE_RATE removed — device native rates read at runtime
# SK200 loopback = 44100 Hz, EMEET mic = 48000 Hz

# Channels from WASAPI loopback source (stereo)
SOURCE_CHANNELS = 2

# Target sample rate for Whisper (required)
WHISPER_SAMPLE_RATE = 16000

# Chunk duration in seconds fed to each capture queue
CHUNK_DURATION_SECONDS = 5

# --- Queue Management ---
MAX_QUEUE_SIZE = 4

# --- Speaker Labels ---
SPEAKER_LOOPBACK = "Guest"   # System audio — the interviewer
SPEAKER_MIC = "Host"         # Microphone — you

# --- Output ---
SESSIONS_DIR = "sessions"
JSON_INDENT = 2

# --- Whisper (Phase 2+) ---
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = "en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
'@ | Set-Content .\config.py