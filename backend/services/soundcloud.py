"""
SoundCloud search service.

SoundCloud is the best source for unofficial remixes, DJ edits, and
sped-up versions that were never released on DSPs. Many TikTok sounds
originate here.

Uses the public SoundCloud API (client_id based, no OAuth required for search).
"""
import httpx
import structlog
from dataclasses import dataclass

log = structlog.get_logger(__name__)

SOUNDCLOUD_API_BASE = "https://api-v2.soundcloud.com"


@dataclass
class SoundCloudTrack:
    track_id: int = 0
    title: str = ""
    artist: str = ""
    genre: str = ""
    duration_ms: int = 0
    play_count: int = 0
    like_count: int = 0
    artwork_url: str = ""
    permalink_url: str = ""
    stream_url: str = ""
    created_at: str = ""
    is_verified_artist: bool = False
    confidence: float = 0.0


async def search(queries: list[dict], settings) -> list[SoundCloudTrack]:
    """Search SoundCloud for matching tracks."""
    if not settings.soundcloud_client_id:
        log.warning("soundcloud_not_configured")
        return []

    results: list[SoundCloudTrack] = []
    seen_ids: set[int] = set()

    for query in queries:
        if query.get("type") not in ("text",):
            continue  # SoundCloud only supports text search

        tracks = await _search_tracks(
            query.get("value", query.get("title", "")),
            settings,
        )
        for track in tracks:
            if track.track_id not in seen_ids:
                seen_ids.add(track.track_id)
                track.confidence = min(1.0, track.confidence + query.get("confidence_bonus", 0))
                results.append(track)

    results.sort(key=lambda t: t.confidence, reverse=True)
    return results[:5]


async def _search_tracks(query: str, settings) -> list[SoundCloudTrack]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{SOUNDCLOUD_API_BASE}/search/tracks",
            params={
                "q": query,
                "client_id": settings.soundcloud_client_id,
                "limit": 5,
                "offset": 0,
            },
        )

    if resp.status_code != 200:
        log.error("soundcloud_search_failed", status=resp.status_code)
        return []

    data = resp.json()
    tracks = []

    for i, item in enumerate(data.get("collection", [])):
        user = item.get("user", {})
        is_verified = user.get("verified", False)

        base_conf = 0.5 - (i * 0.05)
        if is_verified:
            base_conf += 0.1

        title = (item.get("title", "")).lower()
        if any(kw in title for kw in ["sped up", "speed up", "nightcore", "remix"]):
            base_conf -= 0.05  # slight penalty for obvious edits (we want originals)

        tracks.append(SoundCloudTrack(
            track_id=item.get("id", 0),
            title=item.get("title", ""),
            artist=user.get("username", ""),
            genre=item.get("genre", ""),
            duration_ms=item.get("duration", 0),
            play_count=item.get("playback_count", 0),
            like_count=item.get("likes_count", 0),
            artwork_url=(item.get("artwork_url") or "").replace("large", "t300x300"),
            permalink_url=item.get("permalink_url", ""),
            created_at=item.get("created_at", ""),
            is_verified_artist=is_verified,
            confidence=max(0.1, base_conf),
        ))

    return tracks
