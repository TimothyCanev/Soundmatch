"""
Feature extractor.

Extracts musical features from preprocessed audio:
- BPM (tempo)
- Musical key + mode (major/minor)
- Chroma features (12-dim pitch class profile)
- Spectral centroid (brightness)
- Pitch shift detection vs reference

These features are used to:
1. Normalise sped-up / pitch-shifted edits back to their original
2. Drive similarity search when fingerprinting fails
3. Confirm a candidate match found via platform search
"""
import numpy as np
import librosa
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger(__name__)

# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class AudioFeatures:
    bpm: float = 0.0
    key: str = "unknown"
    mode: str = "unknown"        # major / minor
    key_confidence: float = 0.0
    chroma_mean: list[float] = field(default_factory=list)
    spectral_centroid_mean: float = 0.0
    duration: float = 0.0

    # Edit-detection fields (filled by detect_edits)
    estimated_speed_factor: float = 1.0   # 1.15 means sped up 15%
    estimated_pitch_shift: int = 0        # semitones
    normalised_bpm: float = 0.0           # bpm / speed_factor


def extract_features(y: np.ndarray, sr: int) -> AudioFeatures:
    """
    Extract full musical feature set from an audio array.
    This runs synchronously — call from a thread pool in async contexts.
    """
    features = AudioFeatures()
    features.duration = float(len(y) / sr)

    log.info("extracting_features", duration=features.duration)

    # ── BPM ──────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.bpm = float(np.atleast_1d(tempo)[0])

    # ── Chroma / Key ─────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # shape (12,)
    features.chroma_mean = chroma_mean.tolist()

    key_idx, mode, confidence = _detect_key(chroma_mean)
    features.key = _NOTE_NAMES[key_idx]
    features.mode = mode
    features.key_confidence = confidence

    # ── Spectral centroid ─────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.spectral_centroid_mean = float(centroid.mean())

    features.normalised_bpm = features.bpm  # will be updated by detect_edits

    log.info(
        "features_extracted",
        bpm=features.bpm,
        key=f"{features.key} {features.mode}",
        key_confidence=round(features.key_confidence, 3),
    )
    return features


def detect_edits(features: AudioFeatures, reference_bpm: float | None = None) -> AudioFeatures:
    """
    Detect if the clip is a sped-up or pitch-shifted edit.

    If reference_bpm is provided (from a fingerprint match or platform API),
    we compare against it. Otherwise we use heuristics based on common
    TikTok speed factors (1.1x, 1.15x, 1.25x, 1.5x).
    """
    common_speed_factors = [0.85, 0.9, 1.0, 1.1, 1.15, 1.25, 1.5]

    if reference_bpm and reference_bpm > 0 and features.bpm > 0:
        raw_ratio = features.bpm / reference_bpm
        # Snap to nearest common factor within 5% tolerance
        closest = min(common_speed_factors, key=lambda f: abs(f - raw_ratio))
        if abs(closest - raw_ratio) < 0.05:
            features.estimated_speed_factor = closest
        else:
            features.estimated_speed_factor = raw_ratio
    else:
        # Heuristic: typical TikTok sped-up tracks are 140-180 BPM
        # but originals are 90-130. If detected BPM > 150, suspect sped-up.
        if features.bpm > 150:
            features.estimated_speed_factor = 1.15  # most common TikTok speed
        else:
            features.estimated_speed_factor = 1.0

    features.normalised_bpm = features.bpm / features.estimated_speed_factor

    # Pitch shift: estimate as semitones (approximate from spectral centroid shift)
    # In a real v2 system this would use pyin pitch tracking
    features.estimated_pitch_shift = 0

    log.info(
        "edit_detection",
        speed_factor=features.estimated_speed_factor,
        normalised_bpm=round(features.normalised_bpm, 1),
        pitch_shift=features.estimated_pitch_shift,
    )
    return features


def chroma_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two chroma vectors.
    Returns 0.0–1.0, where 1.0 is identical pitch class profiles.
    """
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ─── Internal helpers ─────────────────────────────────────────────

def _detect_key(chroma_mean: np.ndarray) -> tuple[int, str, float]:
    """
    Krumhansl-Schmuckler key-finding algorithm.
    Returns (key_index 0-11, mode, confidence 0-1).
    """
    major_scores = []
    minor_scores = []

    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        major_scores.append(float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]))
        minor_scores.append(float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]))

    best_major_idx = int(np.argmax(major_scores))
    best_minor_idx = int(np.argmax(minor_scores))
    best_major_score = major_scores[best_major_idx]
    best_minor_score = minor_scores[best_minor_idx]

    if best_major_score >= best_minor_score:
        key_idx = best_major_idx
        mode = "major"
        # Confidence: gap between best and second-best
        sorted_scores = sorted(major_scores, reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) * 2
    else:
        key_idx = best_minor_idx
        mode = "minor"
        sorted_scores = sorted(minor_scores, reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) * 2

    confidence = float(np.clip(confidence, 0, 1))
    return key_idx, mode, confidence
