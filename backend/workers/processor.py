"""
Job processor.

The core async pipeline that runs for every identify request:
1. Analyse audio (fingerprint + features)
2. Search all platforms in parallel
3. Rank and return results

Uses an in-memory job store for the MVP. Replace with Redis + BullMQ
for production multi-worker deployments.
"""
import asyncio
import os
import uuid
import time
from typing import Optional
import structlog

from audio_engine import analyse
from audio_engine.audio_matcher import find_by_artist
from services import (
    spotify_search, youtube_search, soundcloud_search,
    rank_results, download_audio, MatchResult,
)
from api.models import JobStatus, MatchType

log = structlog.get_logger(__name__)

# In-memory job store (replace with Redis in production)
_jobs: dict[str, dict] = {}


def create_job() -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "status": JobStatus.queued,
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


async def process_file(job_id: str, file_path: str, settings) -> None:
    """Process an uploaded file. Runs as a background task."""
    _update_job(job_id, status=JobStatus.processing)
    temp_files = [file_path]

    try:
        result = await _run_pipeline(file_path, settings)
        _update_job(job_id, status=JobStatus.complete, result=result)

    except Exception as e:
        log.error("job_failed", job_id=job_id, error=str(e), exc_info=True)
        _update_job(job_id, status=JobStatus.failed, error=str(e))

    finally:
        for path in temp_files:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


async def process_url(job_id: str, url: str, settings) -> None:
    """Download a URL then process it. Runs as a background task."""
    _update_job(job_id, status=JobStatus.processing)
    downloaded_path = None

    try:
        downloaded_path = await download_audio(url, settings.max_clip_duration_seconds)
        result = await _run_pipeline(downloaded_path, settings, url=url)
        _update_job(job_id, status=JobStatus.complete, result=result)

    except ValueError as e:
        # User error (bad URL, too long, etc.)
        _update_job(job_id, status=JobStatus.failed, error=str(e))

    except Exception as e:
        log.error("url_job_failed", job_id=job_id, url=url, error=str(e), exc_info=True)
        _update_job(job_id, status=JobStatus.failed, error="Processing failed. Please try again.")

    finally:
        if downloaded_path and os.path.exists(downloaded_path):
            try:
                os.unlink(downloaded_path)
            except OSError:
                pass


async def _run_pipeline(file_path: str, settings, url: str = "") -> dict:
    """
    Core pipeline: audio analysis → parallel platform search → ranking.
    """
    # Step 1: Audio engine (fingerprint + features)
    engine_result = await analyse(file_path, settings)
# Extract TikTok author for audio matching fallback
    if "tiktok.com" in url or "instagram.com" in url:
        try:
            async with httpx.AsyncClient(timeout=10) as r:
                resp = await r.post("https://www.tikwm.com/api/", data={"url": url}, headers={"User-Agent": "Mozilla/5.0"})
                d = resp.json().get("data", {})
                engine_result.tiktok_author = d.get("music_info", {}).get("author", "") or d.get("author", "")
        except Exception:
            engine_result.tiktok_author = ""
    else:
        engine_result.tiktok_author = ""

    # Step 2: Platform searches in parallel
    queries = engine_result.search_queries

    spotify_task = asyncio.create_task(spotify_search(queries, settings))
    youtube_task = asyncio.create_task(youtube_search(queries, settings))
    soundcloud_task = asyncio.create_task(soundcloud_search(queries, settings))

    spotify_tracks, youtube_videos, soundcloud_tracks = await asyncio.gather(
        spotify_task, youtube_task, soundcloud_task,
        return_exceptions=True,
    )

    # Handle any platform errors gracefully
    if isinstance(spotify_tracks, Exception):
        log.warning("spotify_failed", error=str(spotify_tracks))
        spotify_tracks = []
    if isinstance(youtube_videos, Exception):
        log.warning("youtube_failed", error=str(youtube_videos))
        youtube_videos = []
    if isinstance(soundcloud_tracks, Exception):
        log.warning("soundcloud_failed", error=str(soundcloud_tracks))
        soundcloud_tracks = []

    # Step 3: Rank results
    ranked = rank_results(
        engine_result.fingerprint,
        spotify_tracks,
        youtube_videos,
        soundcloud_tracks,
        engine_result,
    )

    # If fingerprint failed, try audio-to-audio matching via YouTube
    tiktok_author = getattr(engine_result, 'tiktok_author', '')
    if not engine_result.fingerprint.matched and tiktok_author:
        audio_matches = await find_by_artist(
            tiktok_author,
            file_path,
            engine_result.features,
            settings,
        )
        for m in audio_matches:
            from services.ranker import MatchResult, PlatformLinks, EditInfo
            ranked.append(MatchResult(
                rank=0,
                title=m.title,
                artist=m.artist,
                confidence=m.overall_score,
                confidence_label=m.confidence_label,
                platforms=PlatformLinks(youtube=m.youtube_url),
                edit_info=EditInfo(),
                artwork_url=m.thumbnail_url,
            ))

    # Determine match type
    if engine_result.fingerprint.matched and not engine_result.edit_detected:
        match_type = MatchType.exact
    elif engine_result.fingerprint.matched and engine_result.edit_detected:
        match_type = MatchType.edit_detected
    elif ranked:
        match_type = MatchType.similar
    else:
        match_type = MatchType.no_match

    fp = engine_result.fingerprint
    features = engine_result.features

    return {
        "match_type": match_type,
        "results": ranked,
        "audio_features": {
            "bpm": round(features.bpm, 1),
            "key": features.key,
            "mode": features.mode,
            "key_confidence": round(features.key_confidence, 3),
            "estimated_speed_factor": features.estimated_speed_factor,
            "estimated_pitch_shift": features.estimated_pitch_shift,
        },
        "processing_time_ms": engine_result.processing_time_ms,
    }


def _update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)
