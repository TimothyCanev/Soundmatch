"""API data models."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class MatchType(str, Enum):
    exact = "exact"
    edit_detected = "edit_detected"
    similar = "similar"
    no_match = "no_match"


class EditInfoModel(BaseModel):
    detected: bool
    edit_type: str
    speed_factor: float
    pitch_shift_semitones: int


class PlatformLinksModel(BaseModel):
    spotify: Optional[str] = None
    youtube: Optional[str] = None
    soundcloud: Optional[str] = None
    apple_music: Optional[str] = None


class MatchResultModel(BaseModel):
    rank: int
    title: str
    artist: str
    album: str = ""
    release_year: str = ""
    confidence: float
    confidence_label: str
    platforms: PlatformLinksModel
    edit_info: EditInfoModel
    artwork_url: str = ""
    preview_url: str = ""
    duration_ms: int = 0


class AudioFeaturesModel(BaseModel):
    bpm: float
    key: str
    mode: str
    key_confidence: float
    estimated_speed_factor: float
    estimated_pitch_shift: int


class IdentifyResponse(BaseModel):
    job_id: str
    status: JobStatus


class ResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    match_type: Optional[MatchType] = None
    results: list[MatchResultModel] = []
    audio_features: Optional[AudioFeaturesModel] = None
    processing_time_ms: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    services: dict = Field(default_factory=dict)
