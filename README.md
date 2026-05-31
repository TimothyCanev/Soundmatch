# SoundMatch 🎵

> Upload any audio clip — TikTok, Instagram, Reel — and find the exact song or closest match across Spotify, YouTube, and SoundCloud. Handles sped-up edits, pitch-shifted versions, and unofficial remixes.

## Project Structure

```
soundmatch/
├── backend/
│   ├── api/            FastAPI REST endpoints
│   ├── audio_engine/   Fingerprinting + melody analysis
│   ├── services/       Platform search (Spotify, YouTube, SoundCloud)
│   ├── workers/        Async job processing
│   └── tests/          Test suite
├── frontend/           Next.js web app
├── infra/              Docker + deployment config
└── docs/               Architecture docs
```

## Quick Start

### Backend
```bash
cd backend
cp .env.example .env   # fill in API keys
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker (full stack)
```bash
docker-compose up
```

## Environment Variables

| Variable | Description |
|---|---|
| `ACR_HOST` | ACRCloud API host |
| `ACR_ACCESS_KEY` | ACRCloud access key |
| `ACR_ACCESS_SECRET` | ACRCloud access secret |
| `AUDD_API_TOKEN` | AudD API token |
| `SPOTIFY_CLIENT_ID` | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify app client secret |
| `YOUTUBE_API_KEY` | YouTube Data API v3 key |
| `SOUNDCLOUD_CLIENT_ID` | SoundCloud app client ID |
| `REDIS_URL` | Redis connection URL |
| `DATABASE_URL` | PostgreSQL connection URL |

## API Reference

### POST /api/v1/identify
Upload a file or URL for identification.

**Request (multipart/form-data)**
- `file` — audio/video file (max 60s, any format)
- `url` — alternatively, a TikTok/YouTube/Instagram URL

**Response**
```json
{
  "job_id": "abc123",
  "status": "queued"
}
```

### GET /api/v1/results/{job_id}
Poll for results.

**Response**
```json
{
  "status": "complete",
  "match_type": "edit_detected",
  "confidence": 0.91,
  "results": [
    {
      "rank": 1,
      "title": "Blinding Lights",
      "artist": "The Weeknd",
      "confidence": 0.91,
      "edit_info": { "type": "sped_up", "speed_factor": 1.15, "pitch_shift": 2 },
      "platforms": {
        "spotify": "https://open.spotify.com/track/...",
        "youtube": "https://youtube.com/watch?v=...",
        "soundcloud": null
      }
    }
  ]
}
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Roadmap

- [x] Phase 1 — Audio engine + fingerprinting
- [x] Phase 2 — FastAPI backend + job queue
- [x] Phase 3 — Web frontend
- [ ] Phase 4 — Mobile apps (React Native)
- [ ] Phase 5 — Browser extension + Pro tier
