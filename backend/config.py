from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    secret_key: str = "dev-secret-change-in-production"
    allowed_origins: str = "http://localhost:3000"
    max_file_size_mb: int = 50
    max_clip_duration_seconds: int = 60

    # ACRCloud
    acr_host: str = ""
    acr_access_key: str = ""
    acr_access_secret: str = ""

    # AudD
    audd_api_token: str = ""

    # Spotify
    spotify_client_id: str = ""
    spotify_client_secret: str = ""

    # YouTube
    youtube_api_key: str = ""

    # SoundCloud
    soundcloud_client_id: str = ""

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite+aiosqlite:///./soundmatch.db"

    # Observability
    sentry_dsn: str = ""
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
