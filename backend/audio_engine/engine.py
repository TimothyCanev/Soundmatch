"""
SoundMatch Audio Engine.

Main orchestrator: takes a file path and returns structured identification
results including edit detection, platform search queries, and confidence scores.

Pipeline:
  1. Preprocess (→ 16kHz mono WAV, best segment)
  2. Fingerprint (ACRCloud + AudD in parallel)
  3. Feature extraction (BPM, key, chroma)
  4. Edit detection (speed/pitch normalisation)
  5. Build search queries for platform services
"""
import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import structlog

from .preprocessor import preprocess, load_audio_array
from .fingerprinter import fingerprint, FingerprintResult
from .features import extract_features, detect_edits, AudioFeatures

log = structlog.get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_engine")


@dataclass
class EngineResult:
    # Fingerprint
    fingerprint: FingerprintResult = field(default_factory=FingerprintResult)

    # Audio features
    features: AudioFeatures = field(default_factory=AudioFeatures)

    # Derived
    edit_detected: bool = False
    edit_type: str = "none"          # "none" | "sped_up" | "pitch_shifted" | "both"
    search_queries: list[dict] = field(default_factory=list)

    # Meta
    processing_time_ms: int = 0
    audio_duration: float = 0.0


async def analyse(file_path: str, settings) -> EngineResult:
    """
    Full audio analysis pipeline.

    Args:
        file_path: path to any audio/video file
        settings: app settings (API keys etc.)

    Returns:
        EngineResult with fingerprint, features, and search queries
    """
    import time
    t0 = time.monotonic()

    result = EngineResult()

    # ── Step 1: Preprocess ────────────────────────────────────────
    log.info("engine_start", file=file_path)
    prep = await preprocess(file_path)
    wav_path = prep["wav_path"]
    result.audio_duration = prep["duration"]

    try:
        # ── Step 2: Fingerprint + Feature extraction (parallel) ───
        loop = asyncio.get_event_loop()

        fingerprint_task = asyncio.create_task(fingerprint(wav_path, settings))

        # Feature extraction is CPU-bound, run in thread pool
        y, sr = await loop.run_in_executor(_executor, load_audio_array_sync, wav_path)
        features_task = loop.run_in_executor(
            _executor, extract_features, y, sr
        )

        fp_result, audio_features = await asyncio.gather(
            fingerprint_task, features_task
        )

        result.fingerprint = fp_result
        result.features = audio_features

        # ── Step 3: Edit detection ────────────────────────────────
        reference_bpm = None
        # If we have a Spotify match, we could look up its BPM — for now
        # we use the heuristic mode
        audio_features = detect_edits(audio_features, reference_bpm)
        result.features = audio_features

        speed = audio_features.estimated_speed_factor
        pitch = audio_features.estimated_pitch_shift

        if abs(speed - 1.0) > 0.05 and abs(pitch) > 0:
            result.edit_detected = True
            result.edit_type = "both"
        elif abs(speed - 1.0) > 0.05:
            result.edit_detected = True
            result.edit_type = "sped_up"
        elif abs(pitch) > 0:
            result.edit_detected = True
            result.edit_type = "pitch_shifted"

        # ── Step 4: Build search queries ──────────────────────────
        result.search_queries = _build_search_queries(fp_result, audio_features)

    finally:
        # Clean up temp WAV
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    result.processing_time_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "engine_complete",
        fingerprint_matched=result.fingerprint.matched,
        edit_detected=result.edit_detected,
        edit_type=result.edit_type,
        processing_ms=result.processing_time_ms,
    )
    return result


def load_audio_array_sync(wav_path: str):
    """Sync wrapper for use in thread pool."""
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    return y, sr


def _build_search_queries(fp: FingerprintResult, features: AudioFeatures) -> list[dict]:
    """
    Build a prioritised list of search queries for platform services.

    Returns queries in order of confidence — exact IDs first,
    then text queries, then feature-based fallbacks.
    """
    queries = []

    if fp.matched:
        # Best case: we have exact IDs
        if fp.spotify_id:
            queries.append({
                "type": "spotify_id",
                "value": fp.spotify_id,
                "confidence_bonus": 0.2,
            })
        if fp.youtube_id:
            queries.append({
                "type": "youtube_id",
                "value": fp.youtube_id,
                "confidence_bonus": 0.2,
            })
        if fp.isrc:
            queries.append({
                "type": "isrc",
                "value": fp.isrc,
                "confidence_bonus": 0.15,
            })
        # Text fallback for platforms that don't support ID lookup
        queries.append({
            "type": "text",
            "value": f"{fp.title} {fp.artist}",
            "title": fp.title,
            "artist": fp.artist,
            "confidence_bonus": 0.0,
        })
    else:
        # Fingerprint missed — search by normalised musical features
        # Include both original and normalised BPM to catch sped-up versions
        queries.append({
            "type": "feature_search",
            "bpm": round(features.normalised_bpm),
            "key": f"{features.key} {features.mode}",
            "confidence_bonus": -0.15,  # less confident without fingerprint
        })

    return queries
