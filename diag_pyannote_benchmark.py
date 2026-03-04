# diag_pyannote_benchmark.py
#
# Benchmarks pyannote.audio speaker diarization on this CPU.
# Runs diarization on a synthetic two-speaker audio sample and reports
# wall time, real-time factor, and whether it is viable for near-real-time use.
#
# Prerequisites:
#   pip install pyannote.audio
#   A HuggingFace token with access to:
#     - pyannote/speaker-diarization-3.1
#     - pyannote/segmentation-3.0
#   Accept model conditions at:
#     https://huggingface.co/pyannote/speaker-diarization-3.1
#     https://huggingface.co/pyannote/segmentation-3.0
#
# Usage:
#   python diag_pyannote_benchmark.py --token YOUR_HF_TOKEN

import argparse
import time
import numpy as np
import torch

SAMPLE_RATE = 16000
TEST_DURATION_SECONDS = 30  # Simulate 30 seconds of interview audio


def generate_test_audio(duration: int, sample_rate: int) -> np.ndarray:
    """Generate synthetic speech-like audio for benchmarking."""
    t = np.linspace(0, duration, duration * sample_rate)
    # Two synthetic voices at different frequencies
    voice1 = np.sin(2 * np.pi * 200 * t) * 0.3
    voice2 = np.sin(2 * np.pi * 300 * t) * 0.3
    # Alternate between voices every 5 seconds
    audio = np.zeros_like(t)
    for i in range(0, duration, 10):
        start = i * sample_rate
        mid = (i + 5) * sample_rate
        end = min((i + 10) * sample_rate, len(audio))
        audio[start:mid] = voice1[start:mid]
        audio[mid:end] = voice2[mid:end]
    return audio.astype(np.float32)


def run_benchmark(token: str) -> None:
    print("=" * 60)
    print("pyannote.audio CPU Benchmark")
    print("=" * 60)
    print(f"Test audio duration: {TEST_DURATION_SECONDS}s")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU")
    print()

    # Load pipeline
    print("Loading pyannote speaker-diarization-3.1 pipeline...")
    t0 = time.time()
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        pipeline.to(torch.device("cpu"))
    except Exception as e:
        print(f"ERROR loading pipeline: {e}")
        print()
        print("Ensure you have:")
        print("  1. pip install pyannote.audio")
        print("  2. Accepted model conditions at huggingface.co")
        print("  3. A valid HuggingFace token")
        return

    load_time = time.time() - t0
    print(f"Pipeline loaded in {load_time:.1f}s")
    print()

    # Generate test audio
    print("Generating test audio...")
    audio = generate_test_audio(TEST_DURATION_SECONDS, SAMPLE_RATE)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    input_data = {"waveform": audio_tensor, "sample_rate": SAMPLE_RATE}

    # Run diarization and measure wall time
    print(f"Running diarization on {TEST_DURATION_SECONDS}s audio sample...")
    t0 = time.time()
    try:
        diarization = pipeline(input_data)
    except Exception as e:
        print(f"ERROR during diarization: {e}")
        return

    processing_time = time.time() - t0
    real_time_factor = processing_time / TEST_DURATION_SECONDS

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Audio duration:      {TEST_DURATION_SECONDS:.1f}s")
    print(f"Processing time:     {processing_time:.1f}s")
    print(f"Real-time factor:    {real_time_factor:.2f}x")
    print()

    if real_time_factor < 0.5:
        verdict = "EXCELLENT — well within real-time budget"
    elif real_time_factor < 1.0:
        verdict = "ACCEPTABLE — viable for near-real-time use"
    elif real_time_factor < 2.0:
        verdict = "MARGINAL — post-session processing only"
    else:
        verdict = "TOO SLOW — not viable on this hardware"

    print(f"Verdict: {verdict}")
    print()

    # Show detected speakers
    print("Detected speaker turns:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"  {speaker}: {turn.start:.1f}s -> {turn.end:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pyannote.audio on CPU")
    parser.add_argument("--token", required=True, help="HuggingFace access token")
    args = parser.parse_args()
    run_benchmark(args.token)
