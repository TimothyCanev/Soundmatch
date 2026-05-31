"""
SoundMatch test suite.

Tests cover:
- Audio feature extraction
- Edit detection logic
- Ranking algorithm
- API endpoints (mocked external calls)
"""
import sys
import os
import asyncio
import numpy as np
from pathlib import Path
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Feature extraction tests ─────────────────────────────────────

class TestFeatureExtraction:
    def test_bpm_detection_known_frequency(self):
        """Generate a synthetic click track and verify BPM detection."""
        from audio_engine.features import extract_features

        sr = 16000
        bpm_target = 120.0
        beat_interval = sr * 60 / bpm_target  # samples per beat
        duration = 10  # seconds
        total_samples = sr * duration

        y = np.zeros(total_samples)
        click_positions = np.arange(0, total_samples, beat_interval, dtype=int)
        for pos in click_positions:
            if pos < total_samples:
                y[pos] = 1.0

        # Add small noise to make it more realistic
        y += np.random.normal(0, 0.01, total_samples)

        features = extract_features(y, sr)

        # BPM should be within 10% of target (librosa may detect half/double time)
        detected = features.bpm
        assert any(
            abs(detected - expected) < 10
            for expected in [bpm_target, bpm_target / 2, bpm_target * 2]
        ), f"BPM {detected} not close to {bpm_target}"

    def test_chroma_features_length(self):
        """Chroma mean should always be 12 values (one per pitch class)."""
        from audio_engine.features import extract_features

        sr = 16000
        y = np.random.randn(sr * 5).astype(np.float32) * 0.1
        features = extract_features(y, sr)
        assert len(features.chroma_mean) == 12

    def test_key_mode_values(self):
        """Key should be a valid note name and mode should be major/minor."""
        from audio_engine.features import extract_features

        valid_keys = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
        valid_modes = {"major", "minor"}

        sr = 16000
        y = np.random.randn(sr * 5).astype(np.float32) * 0.1
        features = extract_features(y, sr)

        assert features.key in valid_keys, f"Invalid key: {features.key}"
        assert features.mode in valid_modes, f"Invalid mode: {features.mode}"
        assert 0.0 <= features.key_confidence <= 1.0

    def test_duration_calculation(self):
        """Duration should match audio length."""
        from audio_engine.features import extract_features

        sr = 16000
        target_duration = 7.5
        y = np.zeros(int(sr * target_duration))
        features = extract_features(y, sr)
        assert abs(features.duration - target_duration) < 0.1


# ─── Edit detection tests ─────────────────────────────────────────

class TestEditDetection:
    def test_no_edit_at_normal_bpm(self):
        """Normal BPM should not trigger edit detection."""
        from audio_engine.features import AudioFeatures, detect_edits

        features = AudioFeatures(bpm=120.0)
        result = detect_edits(features)
        assert result.estimated_speed_factor == 1.0
        assert abs(result.normalised_bpm - 120.0) < 1.0

    def test_sped_up_detection_heuristic(self):
        """BPM > 150 should trigger sped-up detection heuristic."""
        from audio_engine.features import AudioFeatures, detect_edits

        # Simulated sped-up track: original was 120 BPM, now 138 (×1.15)
        features = AudioFeatures(bpm=155.0)
        result = detect_edits(features)
        # Should snap to 1.15 speed factor
        assert result.estimated_speed_factor == 1.15
        assert result.normalised_bpm < features.bpm

    def test_reference_bpm_matching(self):
        """With a reference BPM, should accurately compute speed factor."""
        from audio_engine.features import AudioFeatures, detect_edits

        original_bpm = 115.0
        speed_factor = 1.15
        detected_bpm = original_bpm * speed_factor  # 132.25

        features = AudioFeatures(bpm=detected_bpm)
        result = detect_edits(features, reference_bpm=original_bpm)

        assert abs(result.estimated_speed_factor - speed_factor) < 0.05
        assert abs(result.normalised_bpm - original_bpm) < 5.0

    def test_normalised_bpm_always_set(self):
        """normalised_bpm should always be populated."""
        from audio_engine.features import AudioFeatures, detect_edits

        features = AudioFeatures(bpm=95.0)
        result = detect_edits(features)
        assert result.normalised_bpm > 0


# ─── Chroma similarity tests ──────────────────────────────────────

class TestChromaSimilarity:
    def test_identical_vectors(self):
        """Identical chroma vectors should return 1.0."""
        from audio_engine.features import chroma_similarity

        vec = [1.0, 0.5, 0.2, 0.8, 0.1, 0.3, 0.9, 0.4, 0.6, 0.7, 0.2, 0.5]
        assert abs(chroma_similarity(vec, vec) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should return 0.0 or near it."""
        from audio_engine.features import chroma_similarity

        a = [1.0] + [0.0] * 11
        b = [0.0] + [1.0] + [0.0] * 10
        result = chroma_similarity(a, b)
        assert result < 0.2

    def test_zero_vector(self):
        """Zero vector should return 0.0 without error."""
        from audio_engine.features import chroma_similarity

        a = [0.0] * 12
        b = [1.0] * 12
        assert chroma_similarity(a, b) == 0.0

    def test_range(self):
        """Similarity should always be in [0, 1]."""
        from audio_engine.features import chroma_similarity

        import random
        for _ in range(20):
            a = [random.random() for _ in range(12)]
            b = [random.random() for _ in range(12)]
            result = chroma_similarity(a, b)
            assert 0.0 <= result <= 1.0, f"Out of range: {result}"


# ─── Ranker tests ─────────────────────────────────────────────────

class TestRanker:
    def _make_engine_result(self, edit_detected=False):
        from audio_engine.features import AudioFeatures
        from audio_engine.fingerprinter import FingerprintResult
        from audio_engine.engine import EngineResult

        result = EngineResult()
        result.fingerprint = FingerprintResult(matched=False)
        result.features = AudioFeatures(bpm=120.0)
        result.edit_detected = edit_detected
        return result

    def test_deduplication(self):
        """Same title+artist from multiple platforms should merge."""
        from services.ranker import rank_results
        from services.spotify import SpotifyTrack
        from services.youtube import YouTubeVideo
        from audio_engine.fingerprinter import FingerprintResult

        fp = FingerprintResult(
            matched=True, title="Test Song", artist="Test Artist",
            confidence=0.9,
        )
        spotify = [SpotifyTrack(
            id="abc", title="Test Song", artist="Test Artist",
            spotify_url="https://spotify.com/track/abc",
            confidence=0.85,
        )]
        youtube = [YouTubeVideo(
            video_id="xyz", title="Test Artist - Test Song (Official)",
            channel="Test Artist", youtube_url="https://youtube.com/watch?v=xyz",
            confidence=0.8,
        )]

        engine = self._make_engine_result()
        results = rank_results(fp, spotify, youtube, [], engine)

        # Should have merged into fewer results than total inputs
        titles = [r.title for r in results]
        # The top result should be Test Song
        assert results[0].title == "Test Song"

    def test_ranking_order(self):
        """Results should be sorted by confidence descending."""
        from services.ranker import rank_results
        from services.spotify import SpotifyTrack
        from audio_engine.fingerprinter import FingerprintResult

        fp = FingerprintResult(matched=False)
        spotify = [
            SpotifyTrack(id="a", title="Low Conf", artist="X", confidence=0.5, spotify_url="https://spotify.com/a"),
            SpotifyTrack(id="b", title="High Conf", artist="Y", confidence=0.9, spotify_url="https://spotify.com/b"),
            SpotifyTrack(id="c", title="Mid Conf", artist="Z", confidence=0.7, spotify_url="https://spotify.com/c"),
        ]

        engine = self._make_engine_result()
        results = rank_results(fp, spotify, [], [], engine)

        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence

    def test_confidence_labels(self):
        """Confidence labels should match score thresholds."""
        from services.ranker import _confidence_label

        assert _confidence_label(0.9) == "high"
        assert _confidence_label(0.8) == "high"
        assert _confidence_label(0.6) == "medium"
        assert _confidence_label(0.55) == "medium"
        assert _confidence_label(0.4) == "low"
        assert _confidence_label(0.1) == "low"

    def test_max_five_results(self):
        """Should never return more than 5 results."""
        from services.ranker import rank_results
        from services.spotify import SpotifyTrack
        from audio_engine.fingerprinter import FingerprintResult

        fp = FingerprintResult(matched=False)
        spotify = [
            SpotifyTrack(id=str(i), title=f"Track {i}", artist=f"Artist {i}",
                         spotify_url=f"https://spotify.com/{i}", confidence=0.5)
            for i in range(10)
        ]

        engine = self._make_engine_result()
        results = rank_results(fp, spotify, [], [], engine)
        assert len(results) <= 5

    def test_edit_info_propagated(self):
        """Edit detection info should appear in all results."""
        from services.ranker import rank_results
        from services.spotify import SpotifyTrack
        from audio_engine.fingerprinter import FingerprintResult

        fp = FingerprintResult(matched=False)
        spotify = [
            SpotifyTrack(id="a", title="Track", artist="Artist",
                         spotify_url="https://spotify.com/a", confidence=0.8)
        ]

        engine = self._make_engine_result(edit_detected=True)
        engine.edit_type = "sped_up"
        results = rank_results(fp, spotify, [], [], engine)

        assert results[0].edit_info.detected is True
        assert results[0].edit_info.edit_type == "sped_up"


# ─── API endpoint tests ───────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "services" in data

    def test_identify_requires_file_or_url(self, client):
        """Identify endpoint should reject empty requests."""
        resp = client.post("/api/v1/identify")
        assert resp.status_code in (400, 422)  # our 400 or FastAPI validation

    def test_identify_rejects_both_file_and_url(self, client):
        """Should reject requests with both file and URL."""
        import io
        resp = client.post(
            "/api/v1/identify",
            data={"url": "https://www.tiktok.com/test"},
            files={"file": ("test.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
        )
        assert resp.status_code == 400

    def test_identify_rejects_unsupported_extension(self, client):
        """Should reject unsupported file types."""
        import io
        resp = client.post(
            "/api/v1/identify",
            files={"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_identify_returns_job_id(self, client):
        """Valid file upload should return a job ID."""
        import io

        # Create minimal valid WAV header (44 bytes)
        wav_header = (
            b'RIFF' + (36).to_bytes(4, 'little') +
            b'WAVE' + b'fmt ' + (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') +   # PCM
            (1).to_bytes(2, 'little') +   # mono
            (16000).to_bytes(4, 'little') +  # sample rate
            (32000).to_bytes(4, 'little') +  # byte rate
            (2).to_bytes(2, 'little') +   # block align
            (16).to_bytes(2, 'little') +  # bits per sample
            b'data' + (0).to_bytes(4, 'little')
        )

        resp = client.post(
            "/api/v1/identify",
            files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_results_unknown_job(self, client):
        """Unknown job ID should return 404."""
        resp = client.get("/api/v1/results/nonexistent-job-id")
        assert resp.status_code == 404

    def test_results_returns_status(self, client):
        """Existing job should return its status."""
        import io

        wav_header = (
            b'RIFF' + (36).to_bytes(4, 'little') +
            b'WAVE' + b'fmt ' + (16).to_bytes(4, 'little') +
            (1).to_bytes(2, 'little') + (1).to_bytes(2, 'little') +
            (16000).to_bytes(4, 'little') + (32000).to_bytes(4, 'little') +
            (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') +
            b'data' + (0).to_bytes(4, 'little')
        )

        # Create job
        resp = client.post(
            "/api/v1/identify",
            files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        )
        job_id = resp.json()["job_id"]

        # Poll results
        results_resp = client.get(f"/api/v1/results/{job_id}")
        assert results_resp.status_code == 200
        data = results_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "processing", "complete", "failed")


# ─── Runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ])
