"""
Audio-to-audio matcher.

When fingerprinting fails, this module:
1. Takes the artist name from TikTok metadata
2. Searches YouTube for that artist
3. Downloads snippets via cobalt.tools API (bypasses server IP blocks)
4. Compares chroma + BPM against the original clip
5. Returns matches above a similarity threshold
"""
import asyncio
import os
import sys
import json
import tempfile
import httpx
import numpy as np
import librosa
import structlog
from dataclasses import dataclass

log = structlog.get_logger(__name__)

SIMILARITY_THRESHOLD = 0.60
BPM_TOLERANCE = 0.12
YOUTUBE_SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
SNIPPET_DURATION = 15


@dataclass
class AudioMatch:
    title: str = ""
    artist: str = ""
    youtube_url: str = ""
    youtube_id: str = ""
    thumbnail_url: str = ""
    chroma_similarity: float = 0.0
    bpm_similarity: float = 0.0
    overall_score: float = 0.0
    confidence_label: str = ""


async def find_by_artist(
    artist_name: str,
    original_wav_path: str,
    original_features,
    settings,
    max_candidates: int = 5,
) -> list[AudioMatch]:
    if not artist_name or artist_name.lower().startswith("original sound"):
        log.info("audio_matcher_skip", reason="no useful artist name")
        return []

    log.info("audio_matcher_start", artist=artist_name)

    candidates = await _search_youtube(artist_name, settings, max_candidates)
    if not candidates:
        log.info("audio_matcher_no_candidates", artist=artist_name)
        return []

    log.info("audio_matcher_candidates", count=len(candidates), artist=artist_name)

    matches = []
    for candidate in candidates:
        try:
            match = await _compare_candidate(
                candidate, original_wav_path, original_features, settings
            )
            if match and match.overall_score >= SIMILARITY_THRESHOLD:
                matches.append(match)
                log.info("audio_match_found", title=match.title, score=round(match.overall_score, 3))
        except Exception as e:
            log.warning("candidate_comparison_failed", error=str(e), title=candidate.get("title"))
            continue

    matches.sort(key=lambda m: m.overall_score, reverse=True)
    log.info("audio_matcher_done", matches=len(matches))
    return matches


async def _search_youtube(artist: str, settings, max_results: int) -> list[dict]:
    if not settings.youtube_api_key or settings.youtube_api_key == "your_youtube_data_api_v3_key":
        return await _search_youtube_ytdlp(artist, max_results)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            YOUTUBE_SEARCH_API,
            params={
                "q": artist,
                "type": "video",
                "videoCategoryId": "10",
                "part": "snippet",
                "maxResults": max_results,
                "key": settings.youtube_api_key,
            },
        )

    if resp.status_code != 200:
        log.warning("youtube_search_api_failed", status=resp.status_code)
        return await _search_youtube_ytdlp(artist, max_results)

    items = resp.json().get("items", [])
    candidates = []
    for item in items:
        video_id = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})
        if video_id:
            candidates.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            })
    return candidates


async def _search_youtube_ytdlp(artist: str, max_results: int) -> list[dict]:
    python_exe = sys.executable
    search_query = f"ytsearch{max_results}:{artist}"

    cmd = [
        python_exe, "-m", "yt_dlp",
        "--no-playlist",
        "--dump-json",
        "--no-warnings",
        "--flat-playlist",
        search_query,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    candidates = []
    for line in stdout.decode(errors="replace").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            video_id = data.get("id", "")
            if video_id:
                candidates.append({
                    "video_id": video_id,
                    "title": data.get("title", ""),
                    "channel": data.get("channel", data.get("uploader", "")),
                    "thumbnail": data.get("thumbnail", ""),
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                })
        except Exception:
            continue

    return candidates


async def _compare_candidate(
    candidate: dict,
    original_wav_path: str,
    original_features,
    settings,
) -> AudioMatch | None:
    video_id = candidate["video_id"]
    youtube_url = candidate["youtube_url"]

    log.info("comparing_candidate", title=candidate["title"], video_id=video_id)

    # Use cobalt.tools API to download YouTube audio — bypasses server IP blocks
    tmp_dir = tempfile.mkdtemp(prefix="sm_cmp_")
    actual_path = os.path.join(tmp_dir, "cmp.mp3")

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.cobalt.tools/",
                json={
                    "url": youtube_url,
                    "downloadMode": "audio",
                    "audioFormat": "mp3",
                },
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        if resp.status_code != 200:
            log.warning("cobalt_api_failed", status=resp.status_code, title=candidate["title"])
            return None

        cobalt_data = resp.json()
        audio_url = cobalt_data.get("url")

        if not audio_url:
            log.warning("cobalt_no_url", title=candidate["title"], response=str(cobalt_data)[:200])
            return None

        # Download the audio file
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            audio_resp = await client.get(audio_url)

        if audio_resp.status_code != 200:
            log.warning("cobalt_download_failed", status=audio_resp.status_code, title=candidate["title"])
            return None

        with open(actual_path, "wb") as f:
            f.write(audio_resp.content)

    except Exception as e:
        log.warning("snippet_download_failed", error=str(e), title=candidate["title"])
        return None

    if not os.path.exists(actual_path) or os.path.getsize(actual_path) < 5000:
        log.warning("snippet_too_small", title=candidate["title"])
        return None

    log.info("snippet_downloaded", title=candidate["title"], size_kb=os.path.getsize(actual_path) // 1024)

    try:
        y, sr = librosa.load(actual_path, sr=16000, mono=True, duration=SNIPPET_DURATION)
        if len(y) < sr * 3:
            return None

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        candidate_chroma = chroma.mean(axis=1).tolist()
        chroma_sim = _cosine_similarity(original_features.chroma_mean, candidate_chroma)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        candidate_bpm = float(np.atleast_1d(tempo)[0])
        original_bpm = original_features.bpm

        bpm_ratios = [
            candidate_bpm / original_bpm if original_bpm > 0 else 1,
            (candidate_bpm * 2) / original_bpm if original_bpm > 0 else 1,
            (candidate_bpm / 2) / original_bpm if original_bpm > 0 else 1,
        ]
        best_bpm_ratio = min(bpm_ratios, key=lambda r: abs(r - 1.0))
        bpm_sim = max(0.0, min(1.0, 1.0 - abs(best_bpm_ratio - 1.0) / BPM_TOLERANCE))

        overall = (chroma_sim * 0.75) + (bpm_sim * 0.25)

        log.info(
            "candidate_scores",
            title=candidate["title"],
            chroma=round(chroma_sim, 3),
            bpm_sim=round(bpm_sim, 3),
            overall=round(overall, 3),
            orig_bpm=round(original_bpm, 1),
            cand_bpm=round(candidate_bpm, 1),
        )

        label = "high" if overall >= 0.88 else "medium" if overall >= 0.78 else "low"

        return AudioMatch(
            title=candidate["title"],
            artist=candidate["channel"],
            youtube_url=youtube_url,
            youtube_id=video_id,
            thumbnail_url=candidate.get("thumbnail", ""),
            chroma_similarity=chroma_sim,
            bpm_similarity=bpm_sim,
            overall_score=overall,
            confidence_label=label,
        )

    except Exception as e:
        log.warning("feature_comparison_failed", error=str(e))
        return None

    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))