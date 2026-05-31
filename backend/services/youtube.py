"""
YouTube search service.

Uses YouTube Data API v3 to find official videos and audio uploads.
Best for finding unofficial remixes and edits that didn't make it to DSPs.
"""
import httpx
import structlog
from dataclasses import dataclass

log = structlog.get_logger(__name__)


@dataclass
class YouTubeVideo:
    video_id: str = ""
    title: str = ""
    channel: str = ""
    channel_verified: bool = False
    description: str = ""
    thumbnail_url: str = ""
    youtube_url: str = ""
    view_count: int = 0
    published_at: str = ""
    is_official: bool = False
    confidence: float = 0.0


async def search(queries: list[dict], settings) -> list[YouTubeVideo]:
    """Search YouTube for matching tracks."""
    if not settings.youtube_api_key:
        log.warning("youtube_not_configured")
        return []

    results: list[YouTubeVideo] = []
    seen_ids: set[str] = set()

    for query in queries:
        videos = await _execute_query(query, settings)
        for video in videos:
            if video.video_id not in seen_ids:
                seen_ids.add(video.video_id)
                video.confidence = min(1.0, video.confidence + query.get("confidence_bonus", 0))
                results.append(video)

    results.sort(key=lambda v: v.confidence, reverse=True)
    return results[:5]


async def _execute_query(query: dict, settings) -> list[YouTubeVideo]:
    qtype = query.get("type")

    if qtype == "youtube_id":
        return await _lookup_by_id(query["value"], settings)

    elif qtype in ("text", "isrc"):
        search_term = query.get("value", "")
        if qtype == "text":
            title = query.get("title", "")
            artist = query.get("artist", "")
            search_term = f"{artist} - {title} official" if artist else f"{title} official audio"
        return await _search_videos(search_term, settings)

    return []


async def _lookup_by_id(video_id: str, settings) -> list[YouTubeVideo]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "id": video_id,
                "part": "snippet,statistics",
                "key": settings.youtube_api_key,
            },
        )
    if resp.status_code != 200:
        return []

    items = resp.json().get("items", [])
    return [_parse_video(v, confidence=0.95) for v in items if v]


async def _search_videos(search_query: str, settings) -> list[YouTubeVideo]:
    async with httpx.AsyncClient(timeout=10) as client:
        # First: search
        search_resp = await client.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": search_query,
                "type": "video",
                "videoCategoryId": "10",  # Music category
                "part": "snippet",
                "maxResults": 5,
                "key": settings.youtube_api_key,
            },
        )

    if search_resp.status_code != 200:
        log.error("youtube_search_failed", status=search_resp.status_code)
        return []

    items = search_resp.json().get("items", [])
    videos = []

    for i, item in enumerate(items):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")
        if not video_id:
            continue

        title = snippet.get("title", "").lower()
        channel = snippet.get("channelTitle", "").lower()

        is_official = any(kw in title for kw in [
            "official", "audio", "lyrics", "music video", "mv"
        ]) or any(kw in channel for kw in [
            "official", "music", "records", "vevo"
        ])

        # Confidence: official videos rank higher
        base_conf = 0.7 - (i * 0.05)
        if is_official:
            base_conf += 0.1

        videos.append(YouTubeVideo(
            video_id=video_id,
            title=snippet.get("title", ""),
            channel=snippet.get("channelTitle", ""),
            description=snippet.get("description", "")[:200],
            thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            youtube_url=f"https://www.youtube.com/watch?v={video_id}",
            published_at=snippet.get("publishedAt", ""),
            is_official=is_official,
            confidence=base_conf,
        ))

    return videos


def _parse_video(data: dict, confidence: float = 0.9) -> YouTubeVideo:
    snippet = data.get("snippet", {})
    stats = data.get("statistics", {})
    video_id = data.get("id", "")

    title = snippet.get("title", "").lower()
    channel = snippet.get("channelTitle", "").lower()
    is_official = any(kw in title for kw in ["official", "audio", "lyrics"]) or \
                  any(kw in channel for kw in ["official", "vevo"])

    return YouTubeVideo(
        video_id=video_id,
        title=snippet.get("title", ""),
        channel=snippet.get("channelTitle", ""),
        description=snippet.get("description", "")[:200],
        thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        youtube_url=f"https://www.youtube.com/watch?v={video_id}",
        published_at=snippet.get("publishedAt", ""),
        view_count=int(stats.get("viewCount", 0)),
        is_official=is_official,
        confidence=confidence,
    )
