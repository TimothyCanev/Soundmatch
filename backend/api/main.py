"""
SoundMatch API — FastAPI application.

Endpoints:
  GET  /health                   — health check
  POST /api/v1/identify          — submit audio file or URL
  GET  /api/v1/results/{job_id}  — poll for results
"""
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import sys
import asyncio
import tempfile
import aiofiles
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from api.models import (
    IdentifyResponse, ResultsResponse, HealthResponse,
    JobStatus, MatchResultModel, AudioFeaturesModel, EditInfoModel, PlatformLinksModel,
)
from workers.processor import create_job, get_job, process_file, process_url

log = structlog.get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".webm", ".mov", ".aac", ".flac"}
MAX_FILE_BYTES = settings.max_file_size_mb * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("soundmatch_api_starting", env=settings.app_env)
    yield
    log.info("soundmatch_api_shutdown")


app = FastAPI(
    title="SoundMatch API",
    description="Identify songs from short audio clips, including sped-up TikTok edits.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        services={
            "acrcloud": bool(settings.acr_access_key),
            "audd": bool(settings.audd_api_token),
            "spotify": bool(settings.spotify_client_id),
            "youtube": bool(settings.youtube_api_key),
            "soundcloud": bool(settings.soundcloud_client_id),
        },
    )


# ─── Identify ─────────────────────────────────────────────────────

@app.post("/api/v1/identify", response_model=IdentifyResponse, tags=["Identify"])
async def identify(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
):
    """
    Submit an audio/video file or social media URL for identification.

    - **file**: upload an audio/video file (mp3, mp4, wav, etc.)
    - **url**: TikTok, Instagram, YouTube, or other social media URL

    Returns a job_id. Poll `/api/v1/results/{job_id}` for results.
    """
    if not file and not url:
        raise HTTPException(400, "Provide either a file upload or a URL.")
    if file and url:
        raise HTTPException(400, "Provide either a file or a URL, not both.")

    job_id = create_job()

    if file:
        # Validate file
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # Save to temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="sm_upload_")
        os.close(tmp_fd)

        try:
            content = await file.read()
            if len(content) > MAX_FILE_BYTES:
                os.unlink(tmp_path)
                raise HTTPException(
                    413,
                    f"File too large. Maximum size is {settings.max_file_size_mb}MB.",
                )
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(content)
        except HTTPException:
            raise
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(500, f"Failed to save uploaded file: {e}")

        background_tasks.add_task(process_file, job_id, tmp_path, settings)
        log.info("job_queued_file", job_id=job_id, filename=file.filename)

    else:
        background_tasks.add_task(process_url, job_id, url, settings)
        log.info("job_queued_url", job_id=job_id, url=url)

    return IdentifyResponse(job_id=job_id, status=JobStatus.queued)


# ─── Results ──────────────────────────────────────────────────────

@app.get("/api/v1/results/{job_id}", response_model=ResultsResponse, tags=["Identify"])
async def get_results(job_id: str):
    """
    Poll for results of an identification job.

    Status values:
    - `queued` — waiting to start
    - `processing` — currently analysing
    - `complete` — results ready
    - `failed` — processing error (see `error` field)
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    status = job["status"]

    if status in (JobStatus.queued, JobStatus.processing):
        return ResultsResponse(job_id=job_id, status=status)

    if status == JobStatus.failed:
        return ResultsResponse(
            job_id=job_id,
            status=status,
            error=job.get("error", "Unknown error"),
        )

    # Complete — build response
    result = job["result"]
    ranked = result["results"]
    features_data = result.get("audio_features", {})

    match_results = []
    for r in ranked:
        platforms = r.platforms
        edit = r.edit_info
        match_results.append(MatchResultModel(
            rank=r.rank,
            title=r.title,
            artist=r.artist,
            album=r.album,
            release_year=r.release_year,
            confidence=round(r.confidence, 3),
            confidence_label=r.confidence_label,
            platforms=PlatformLinksModel(
                spotify=platforms.spotify or None,
                youtube=platforms.youtube or None,
                soundcloud=platforms.soundcloud or None,
            ),
            edit_info=EditInfoModel(
                detected=edit.detected,
                edit_type=edit.edit_type,
                speed_factor=edit.speed_factor,
                pitch_shift_semitones=edit.pitch_shift_semitones,
            ),
            artwork_url=r.artwork_url or "",
            preview_url=r.preview_url or "",
            duration_ms=r.duration_ms,
        ))

    audio_features = None
    if features_data:
        audio_features = AudioFeaturesModel(
            bpm=features_data.get("bpm", 0),
            key=features_data.get("key", ""),
            mode=features_data.get("mode", ""),
            key_confidence=features_data.get("key_confidence", 0),
            estimated_speed_factor=features_data.get("estimated_speed_factor", 1.0),
            estimated_pitch_shift=features_data.get("estimated_pitch_shift", 0),
        )

    return ResultsResponse(
        job_id=job_id,
        status=status,
        match_type=result["match_type"],
        results=match_results,
        audio_features=audio_features,
        processing_time_ms=result.get("processing_time_ms", 0),
    )


# ─── Error handlers ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=str(request.url), error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
