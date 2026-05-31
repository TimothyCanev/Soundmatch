"""
Audio fingerprinter.

Submits audio to ACRCloud and AudD for exact-match identification.
Both APIs run in parallel; we return whichever responds first with
a confident match. Falls back gracefully if no exact match is found.
"""
import asyncio
import base64
import hashlib
import hmac
import time
import os
from pathlib import Path
import httpx
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger(__name__)


@dataclass
class FingerprintResult:
    matched: bool = False
    source: str = ""          # "acrcloud" | "audd" | "none"
    title: str = ""
    artist: str = ""
    album: str = ""
    release_year: str = ""
    isrc: str = ""
    spotify_id: str = ""
    youtube_id: str = ""
    confidence: float = 0.0
    raw: dict = field(default_factory=dict)


async def fingerprint(wav_path: str, settings) -> FingerprintResult:
    """
    Run fingerprinting against ACRCloud and AudD in parallel.
    Returns the best result.
    """
    tasks = []

    if settings.acr_access_key:
        tasks.append(_query_acrcloud(wav_path, settings))
    if settings.audd_api_token:
        tasks.append(_query_audd(wav_path, settings))

    if not tasks:
        log.warning("no_fingerprint_api_configured")
        return FingerprintResult(matched=False, source="none")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Pick the best non-error result
    best = FingerprintResult(matched=False, source="none")
    for r in results:
        if isinstance(r, Exception):
            log.warning("fingerprint_api_error", error=str(r))
            continue
        if r.matched and r.confidence > best.confidence:
            best = r

    log.info(
        "fingerprint_complete",
        matched=best.matched,
        source=best.source,
        title=best.title,
        confidence=best.confidence,
    )
    return best


# ─── ACRCloud ─────────────────────────────────────────────────────

async def _query_acrcloud(wav_path: str, settings) -> FingerprintResult:
    """Query ACRCloud identify endpoint with HMAC-SHA1 auth."""
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(time.time())

    string_to_sign = "\n".join([
        http_method, http_uri, settings.acr_access_key,
        data_type, signature_version, timestamp,
    ])

    signature = base64.b64encode(
        hmac.new(
            settings.acr_access_secret.encode(),
            string_to_sign.encode(),
            "sha1",
        ).digest()
    ).decode()

    wav_bytes = Path(wav_path).read_bytes()

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"https://{settings.acr_host}/v1/identify",
            data={
                "access_key": settings.acr_access_key,
                "sample_bytes": str(len(wav_bytes)),
                "timestamp": timestamp,
                "signature": signature,
                "data_type": data_type,
                "signature_version": signature_version,
            },
            files={"sample": ("sample.wav", wav_bytes, "audio/wav")},
        )

    data = response.json()
    log.debug("acrcloud_raw_response", status=data.get("status"))

    status = data.get("status", {})
    if status.get("code") != 0:
        return FingerprintResult(matched=False, source="acrcloud", raw=data)

    music = data.get("metadata", {}).get("music", [])
    if not music:
        return FingerprintResult(matched=False, source="acrcloud", raw=data)

    top = music[0]
    score = top.get("score", 0) / 100.0  # ACR gives 0-100

    external_ids = top.get("external_ids", {})
    external_metadata = top.get("external_metadata", {})
    spotify_meta = external_metadata.get("spotify", {}).get("track", {})
    youtube_meta = external_metadata.get("youtube", {})

    artists = top.get("artists", [{}])
    artist_name = artists[0].get("name", "") if artists else ""

    return FingerprintResult(
        matched=score > 0.5,
        source="acrcloud",
        title=top.get("title", ""),
        artist=artist_name,
        album=top.get("album", {}).get("name", ""),
        release_year=str(top.get("release_date", ""))[:4],
        isrc=external_ids.get("isrc", ""),
        spotify_id=spotify_meta.get("id", ""),
        youtube_id=youtube_meta.get("vid", ""),
        confidence=score,
        raw=data,
    )


# ─── AudD ─────────────────────────────────────────────────────────

async def _query_audd(wav_path: str, settings) -> FingerprintResult:
    """Query AudD API — good secondary source, especially for recent releases."""
    wav_bytes = Path(wav_path).read_bytes()

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://api.audd.io/",
            data={
                "api_token": settings.audd_api_token,
                "return": "spotify,apple_music,deezer",
            },
            files={"file": ("sample.wav", wav_bytes, "audio/wav")},
        )

    data = response.json()
    log.debug("audd_raw_response", status=data.get("status"))

    if data.get("status") != "success" or not data.get("result"):
        return FingerprintResult(matched=False, source="audd", raw=data)

    result = data["result"]
    spotify_id = ""
    spotify_data = result.get("spotify")
    if spotify_data:
        spotify_id = spotify_data.get("id", "")

    return FingerprintResult(
        matched=True,
        source="audd",
        title=result.get("title", ""),
        artist=result.get("artist", ""),
        album=result.get("album", ""),
        release_year=str(result.get("release_date", ""))[:4],
        isrc=result.get("isrc", ""),
        spotify_id=spotify_id,
        confidence=0.85,  # AudD doesn't return a score; 0.85 is a reasonable default
        raw=data,
    )
