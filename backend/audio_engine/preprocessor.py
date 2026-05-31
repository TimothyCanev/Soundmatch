"""
Audio preprocessor.

Converts any input (file path or downloaded clip) to mono 16kHz WAV —
the standard format required by fingerprinting APIs and librosa analysis.
Also extracts a clean 15-second segment best suited for identification.
"""
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import librosa
import structlog

log = structlog.get_logger(__name__)

SAMPLE_RATE = 16000
TARGET_DURATION = 15  # seconds to extract for fingerprinting


async def preprocess(input_path: str) -> dict:
    """
    Convert input audio/video to analysis-ready format.

    Returns:
        {
            "wav_path": str,          # path to 16kHz mono WAV
            "duration": float,         # total duration in seconds
            "sample_rate": int,
            "segment_start": float,    # where the extracted segment begins
        }
    """
    input_path = str(input_path)
    log.info("preprocessing_audio", input=input_path)

    # Step 1: probe duration
    duration = await _probe_duration(input_path)
    log.info("probed_duration", seconds=duration)

    # Step 2: find best segment (loudest / most melodic part)
    segment_start = _find_best_segment(input_path, duration)

    # Step 3: extract and convert to 16kHz mono WAV
    wav_path = await _extract_wav(input_path, segment_start, TARGET_DURATION)

    return {
        "wav_path": wav_path,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "segment_start": segment_start,
    }


async def _probe_duration(path: str) -> float:
    """Use ffprobe to get duration without loading the whole file."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 30.0  # fallback


def _find_best_segment(path: str, total_duration: float) -> float:
    """
    Find the most melodically rich segment for fingerprinting.
    Strategy: load a downsampled mono version, compute RMS energy in
    windows, skip the first 5s (often intro silence), pick the
    highest-energy window.
    """
    if total_duration <= TARGET_DURATION:
        return 0.0

    try:
        # Load at low SR for fast scanning
        y, sr = librosa.load(path, sr=8000, mono=True, duration=min(total_duration, 120))

        window = int(sr * TARGET_DURATION)
        hop = int(sr * 5)  # 5s hops
        skip_start = int(sr * 5)  # skip first 5s

        best_rms = -1.0
        best_start_sample = skip_start

        for i in range(skip_start, len(y) - window, hop):
            segment = y[i:i + window]
            rms = float(np.sqrt(np.mean(segment ** 2)))
            if rms > best_rms:
                best_rms = rms
                best_start_sample = i

        return float(best_start_sample / sr)

    except Exception as e:
        log.warning("segment_detection_failed", error=str(e))
        # Fall back to 10% in — usually past any intro
        return max(0.0, total_duration * 0.1)


async def _extract_wav(input_path: str, start: float, duration: float) -> str:
    """Extract a segment as 16kHz mono WAV using ffmpeg."""
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="sm_")
    os.close(out_fd)
    
    cmd = [
        "ffmpeg", "-y",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-ss", str(start),
        "-t", str(duration),
        "-i", input_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-vn",
        "-f", "wav",
        out_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()[-300:]}")

    log.info("wav_extracted", path=out_path, start=start, duration=duration)
    return out_path


async def load_audio_array(wav_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV into a numpy array for feature extraction."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    return y, sr
