"""
Spotify search service.

Uses the Spotify Web API (Client Credentials flow) to:
1. Look up tracks by Spotify ID (from fingerprint)
2. Search by ISRC (most precise text-based lookup)
3. Search by title + artist
4. Search by audio features (BPM + key) as last resort
"""
import asyncio
import time
import httpx
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger(__name__)

_token_cache: dict = {"token": None, "expires_at": 0.0}


@dataclass
class SpotifyTrack:
    id: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    release_year: str = ""
    duration_ms: int = 0
    preview_url: str = ""
    spotify_url: str = ""
    popularity: int = 0
    isrc: str = ""
    bpm: float = 0.0
    key: int = -1
    mode: int = -1
    confidence: float = 0.0


async def search(queries: list[dict], settings) -> list[SpotifyTrack]:
    """
    Execute Spotify searches for a list of queries.
    Returns ranked, deduplicated list of tracks.
    """
    if not settings.spotify_client_id or not settings.spotify_client_secret:
        log.warning("spotify_not_configured")
        return []

    token = await _get_token(settings)
    if not token:
        return []

    results: list[SpotifyTrack] = []
    seen_ids: set[str] = set()

    for query in queries:
        tracks = await _execute_query(query, token, settings)
        for track in tracks:
            if track.id not in seen_ids:
                seen_ids.add(track.id)
                # Apply confidence bonus from query
                track.confidence = min(1.0, track.confidence + query.get("confidence_bonus", 0))
                results.append(track)

    # Sort by confidence desc
    results.sort(key=lambda t: t.confidence, reverse=True)
    return results[:5]


async def _execute_query(query: dict, token: str, settings) -> list[SpotifyTrack]:
    qtype = query.get("type")

    if qtype == "spotify_id":
        return await _lookup_by_id(query["value"], token)

    elif qtype == "isrc":
        return await _search_by_isrc(query["value"], token)

    elif qtype == "text":
        return await _search_by_text(
            query.get("title", ""),
            query.get("artist", ""),
            token,
        )

    elif qtype == "feature_search":
        # Feature-based search: broader text search, then filter by BPM
        return await _search_by_features(
            query.get("bpm", 0),
            query.get("key", ""),
            token,
        )

    return []


async def _lookup_by_id(track_id: str, token: str) -> list[SpotifyTrack]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"https://api.spotify.com/v1/tracks/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 200:
        return []

    return [_parse_track(resp.json(), confidence=0.95)]


async def _search_by_isrc(isrc: str, token: str) -> list[SpotifyTrack]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.spotify.com/v1/search",
            params={"q": f"isrc:{isrc}", "type": "track", "limit": 3},
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 200:
        return []

    items = resp.json().get("tracks", {}).get("items", [])
    return [_parse_track(t, confidence=0.9) for t in items if t]


async def _search_by_text(title: str, artist: str, token: str) -> list[SpotifyTrack]:
    query = f"track:{title} artist:{artist}" if artist else title
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 200:
        return []

    items = resp.json().get("tracks", {}).get("items", [])
    tracks = []
    for i, t in enumerate(items):
        if t:
            # Confidence decreases with rank
            conf = 0.75 - (i * 0.05)
            tracks.append(_parse_track(t, confidence=conf))
    return tracks


async def _search_by_features(bpm: float, key: str, token: str) -> list[SpotifyTrack]:
    """
    Feature-based search: search broadly then filter using audio features.
    NOTE: Spotify deprecated the /audio-features endpoint for new apps in Nov 2024.
    We fall back to a general search here.
    """
    # Without audio features API, we return empty for feature-only search
    # This would be powered by our own vector store in v2
    log.info("feature_search_skipped_no_api")
    return []


def _parse_track(data: dict, confidence: float = 0.7) -> SpotifyTrack:
    artists = data.get("artists", [{}])
    artist_name = ", ".join(a.get("name", "") for a in artists)
    album = data.get("album", {})
    release_date = album.get("release_date", "")

    external_ids = data.get("external_ids", {})
    external_urls = data.get("external_urls", {})

    return SpotifyTrack(
        id=data.get("id", ""),
        title=data.get("name", ""),
        artist=artist_name,
        album=album.get("name", ""),
        release_year=release_date[:4] if release_date else "",
        duration_ms=data.get("duration_ms", 0),
        preview_url=data.get("preview_url") or "",
        spotify_url=external_urls.get("spotify", ""),
        popularity=data.get("popularity", 0),
        isrc=external_ids.get("isrc", ""),
        confidence=confidence,
    )


async def _get_token(settings) -> str | None:
    """Client credentials token with simple in-memory cache."""
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(settings.spotify_client_id, settings.spotify_client_secret),
        )

    if resp.status_code != 200:
        log.error("spotify_token_failed", status=resp.status_code)
        return None

    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600)
    return _token_cache["token"]
