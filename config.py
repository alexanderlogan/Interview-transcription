# config.py
# Central configuration for Interview Transcriber.
# All tuneable constants live here. Do not scatter magic numbers in source files.

# --- Audio Capture ---

# Source sample rate from WASAPI (hardware default for most devices)
SOURCE_SAMPLE_RATE = 48000

# Target sample rate for Whisper (required)
WHISPER_SAMPLE_RATE = 16000

# Channels from WASAPI source (stereo)
SOURCE_CHANNELS = 2

# Chunk duration in seconds fed to each capture queue
CHUNK_DURATION_SECONDS = 5

# Derived: frames per chunk at source rate
FRAMES_PER_CHUNK = SOURCE_SAMPLE_RATE * CHUNK_DURATION_SECONDS

# --- Queue Management ---

# Maximum chunks held in each capture queue before oldest is dropped
# At 5s chunks, 4 items = 20s of backlog tolerance
MAX_QUEUE_SIZE = 4

# --- Speaker Labels ---
# Stream-based deterministic labels (confirmed approach pending Phase 3 benchmark)
SPEAKER_LOOPBACK = "Guest"   # System audio — the interviewer
SPEAKER_MIC = "Host"        # Microphone — you

# --- Output ---

# Directory for session JSON files (relative to project root)
SESSIONS_DIR = "sessions"

# JSON indent for human readability in session files
JSON_INDENT = 2

# --- Whisper (Phase 2+) ---
# Confirmed on this hardware: base = 2.9s per 5s chunk (acceptable)
# Do not change without re-benchmarking on target CPU.
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = "en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
