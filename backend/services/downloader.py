"""
URL downloader.

Handles audio extraction from social media URLs:
- TikTok / Instagram: via tikwm.com proxy API (no auth needed)
- YouTube / others: via yt-dlp
"""
import asyncio
import os
import sys
import tempfile
import httpx
import structlog

log = structlog.get_logger(__name__)

TIKTOK_DOMAINS = ["tiktok.com", "vm.tiktok.com"]
INSTAGRAM_DOMAINS = ["instagram.com", "www.instagram.com"]
TIKWM_API = "https://www.tikwm.com/api/"


async def download_audio(url: str, max_duration: int = 60) -> str:
    """
    Download audio from a social media URL.
    Returns path to downloaded audio file (caller must clean up).
    """
    _validate_url(url)
    url_lower = url.lower()

    # TikTok and Instagram → use tikwm proxy
    if any(d in url_lower for d in TIKTOK_DOMAINS + INSTAGRAM_DOMAINS):
        return await _download_via_tikwm(url)

    # Everything else (YouTube, Twitter etc.) → yt-dlp
    return await _download_via_ytdlp(url)


async def _download_via_tikwm(url: str) -> str:
    """
    Use tikwm.com API to get a direct CDN link, then download the audio.
    Works for TikTok and Instagram Reels without any authentication.
    """
    log.info("tikwm_download_start", url=url)

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: get video info + direct download URL
        resp = await client.post(
            TIKWM_API,
            data={"url": url, "hd": "0"},
            headers={"User-Agent": "Mozilla/5.0"},
        )

    if resp.status_code != 200:
        raise RuntimeError(f"tikwm API error: HTTP {resp.status_code}")

    data = resp.json()
    log.info("tikwm_response", code=data.get("code"), msg=data.get("msg"))

    if data.get("code") != 0:
        raise ValueError(f"Could not fetch TikTok video: {data.get('msg', 'unknown error')}")

    video_data = data.get("data", {})

    # Try to get audio-only first, fall back to video
    audio_url = (
        video_data.get("music_info", {}).get("play")
        or video_data.get("hdplay")
        or video_data.get("play")
    )

    if not audio_url:
        raise RuntimeError("tikwm returned no playable URL")

    # Get song info for logging
    music = video_data.get("music_info", {})
    log.info(
        "tikwm_track_info",
        title=music.get("title", "unknown"),
        author=music.get("author", "unknown"),
    )

    # Step 2: download the actual file
    tmp_dir = tempfile.mkdtemp(prefix="sm_tikwm_")
    out_path = os.path.join(tmp_dir, "audio.mp3")

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(
            audio_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download audio file: HTTP {resp.status_code}")

    with open(out_path, "wb") as f:
        f.write(resp.content)

    size_kb = os.path.getsize(out_path) // 1024
    log.info("tikwm_download_complete", path=out_path, size_kb=size_kb)

    if size_kb < 5:
        raise RuntimeError("Downloaded file is too small — likely an error page")

    return out_path


async def _download_via_ytdlp(url: str) -> str:
    """Download via yt-dlp for YouTube and other platforms."""
    log.info("ytdlp_download_start", url=url)

    tmp_dir = tempfile.mkdtemp(prefix="sm_dl_")
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")
    python_exe = sys.executable

    cmd = [
        python_exe, "-m", "yt_dlp",
        "--no-playlist",
        "--max-downloads", "1",
        "-f", "140/bestaudio/best",
        "--no-warnings",
        "-o", output_template,
        url,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        error_msg = stderr.decode(errors="replace")[-600:]
        files = [f for f in os.listdir(tmp_dir) if not f.endswith(".part")]
        if not files:
            log.error("ytdlp_download_failed", error=error_msg)
            raise RuntimeError(f"Download failed: {error_msg}")

    # Find the downloaded file
    AUDIO_EXTS = {".mp3", ".m4a", ".webm", ".ogg", ".opus", ".mp4", ".wav", ".aac"}
    actual_path = None
    all_files = os.listdir(tmp_dir)

    for f in all_files:
        if not f.endswith(".part") and any(f.endswith(ext) for ext in AUDIO_EXTS):
            actual_path = os.path.join(tmp_dir, f)
            break

    if not actual_path:
        non_partial = [f for f in all_files if not f.endswith(".part")]
        if non_partial:
            actual_path = os.path.join(tmp_dir, non_partial[0])

    if not actual_path:
        raise RuntimeError("Download completed but output file not found.")

    size_kb = os.path.getsize(actual_path) // 1024
    log.info("ytdlp_download_complete", path=actual_path, size_kb=size_kb)
    return actual_path


def _validate_url(url: str) -> None:
    url_lower = url.lower()
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    supported = [
        "tiktok.com", "vm.tiktok.com",
        "instagram.com", "www.instagram.com",
        "youtube.com", "youtu.be", "www.youtube.com",
        "twitter.com", "x.com", "reddit.com",
    ]
    if not any(d in url_lower for d in supported):
        raise ValueError("Unsupported URL. Supported: TikTok, Instagram, YouTube, Twitter/X, Reddit.")