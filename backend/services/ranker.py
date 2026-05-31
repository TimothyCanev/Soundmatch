"""
Match ranker.

Combines results from fingerprinting and all platform searches into
a single ranked, deduplicated list of MatchResult objects.

Scoring weights:
- Fingerprint confidence:    40%
- Platform match rank:       30%
- Official release bonus:    15%
- Popularity signal:         15%
"""
from dataclasses import dataclass, field
import structlog

log = structlog.get_logger(__name__)


@dataclass
class PlatformLinks:
    spotify: str = ""
    youtube: str = ""
    soundcloud: str = ""
    apple_music: str = ""


@dataclass
class EditInfo:
    detected: bool = False
    edit_type: str = "none"         # none | sped_up | pitch_shifted | both
    speed_factor: float = 1.0
    pitch_shift_semitones: int = 0


@dataclass
class MatchResult:
    rank: int = 0
    title: str = ""
    artist: str = ""
    album: str = ""
    release_year: str = ""
    confidence: float = 0.0
    confidence_label: str = ""      # "high" | "medium" | "low"
    platforms: PlatformLinks = field(default_factory=PlatformLinks)
    edit_info: EditInfo = field(default_factory=EditInfo)
    artwork_url: str = ""
    preview_url: str = ""
    duration_ms: int = 0


def rank_results(
    fingerprint_result,
    spotify_tracks: list,
    youtube_videos: list,
    soundcloud_tracks: list,
    engine_result,
) -> list[MatchResult]:
    """
    Merge and rank all results into a final list.
    """
    # Build candidate pool keyed by normalised title+artist
    candidates: dict[str, dict] = {}

    # ── Seed from fingerprint ─────────────────────────────────────
    if fingerprint_result.matched:
        key = _normalise_key(fingerprint_result.title, fingerprint_result.artist)
        candidates[key] = {
            "title": fingerprint_result.title,
            "artist": fingerprint_result.artist,
            "album": fingerprint_result.album,
            "release_year": fingerprint_result.release_year,
            "base_confidence": fingerprint_result.confidence * 0.9,
            "platforms": PlatformLinks(),
            "artwork_url": "",
            "preview_url": "",
            "duration_ms": 0,
        }

    # ── Merge Spotify ─────────────────────────────────────────────
    for track in spotify_tracks:
        key = _normalise_key(track.title, track.artist)
        if key in candidates:
            candidates[key]["base_confidence"] = max(
                candidates[key]["base_confidence"],
                track.confidence,
            )
            candidates[key]["platforms"].spotify = track.spotify_url
            if track.preview_url:
                candidates[key]["preview_url"] = track.preview_url
            if not candidates[key]["artwork_url"]:
                pass  # Spotify doesn't return artwork in track search without album lookup
        else:
            candidates[key] = {
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "release_year": track.release_year,
                "base_confidence": track.confidence,
                "platforms": PlatformLinks(spotify=track.spotify_url),
                "artwork_url": "",
                "preview_url": track.preview_url,
                "duration_ms": track.duration_ms,
            }

    # ── Merge YouTube ─────────────────────────────────────────────
    for video in youtube_videos:
        # YouTube titles are messy; try to match against existing candidates
        matched_key = _fuzzy_match_key(video.title, candidates)
        if matched_key:
            candidates[matched_key]["platforms"].youtube = video.youtube_url
            if video.thumbnail_url and not candidates[matched_key].get("artwork_url"):
                candidates[matched_key]["artwork_url"] = video.thumbnail_url
            # Boost confidence if it's an official video
            if video.is_official:
                candidates[matched_key]["base_confidence"] = min(
                    1.0,
                    candidates[matched_key]["base_confidence"] + 0.05,
                )
        else:
            # Add as new candidate with lower confidence
            key = _normalise_key(video.title, video.channel)
            candidates[key] = {
                "title": video.title,
                "artist": video.channel,
                "album": "",
                "release_year": video.published_at[:4] if video.published_at else "",
                "base_confidence": video.confidence * 0.8,
                "platforms": PlatformLinks(youtube=video.youtube_url),
                "artwork_url": video.thumbnail_url,
                "preview_url": "",
                "duration_ms": 0,
            }

    # ── Merge SoundCloud ──────────────────────────────────────────
    for track in soundcloud_tracks:
        matched_key = _fuzzy_match_key(track.title, candidates)
        if matched_key:
            candidates[matched_key]["platforms"].soundcloud = track.permalink_url
            if track.artwork_url and not candidates[matched_key].get("artwork_url"):
                candidates[matched_key]["artwork_url"] = track.artwork_url
        else:
            key = _normalise_key(track.title, track.artist)
            candidates[key] = {
                "title": track.title,
                "artist": track.artist,
                "album": "",
                "release_year": track.created_at[:4] if track.created_at else "",
                "base_confidence": track.confidence * 0.7,
                "platforms": PlatformLinks(soundcloud=track.permalink_url),
                "artwork_url": track.artwork_url,
                "preview_url": "",
                "duration_ms": track.duration_ms,
            }

    # ── Build final results ───────────────────────────────────────
    edit_info = EditInfo(
        detected=engine_result.edit_detected,
        edit_type=engine_result.edit_type,
        speed_factor=engine_result.features.estimated_speed_factor,
        pitch_shift_semitones=engine_result.features.estimated_pitch_shift,
    )

    results = []
    for candidate in candidates.values():
        conf = float(candidate["base_confidence"])
        # Bonus: has multiple platforms (shows it's a well-known track)
        platforms = candidate["platforms"]
        platform_count = sum([
            bool(platforms.spotify),
            bool(platforms.youtube),
            bool(platforms.soundcloud),
        ])
        if platform_count >= 2:
            conf = min(1.0, conf + 0.05)

        results.append(MatchResult(
            title=candidate["title"],
            artist=candidate["artist"],
            album=candidate.get("album", ""),
            release_year=candidate.get("release_year", ""),
            confidence=conf,
            confidence_label=_confidence_label(conf),
            platforms=platforms,
            edit_info=edit_info,
            artwork_url=candidate.get("artwork_url", ""),
            preview_url=candidate.get("preview_url", ""),
            duration_ms=candidate.get("duration_ms", 0),
        ))

    results.sort(key=lambda r: r.confidence, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    log.info("ranking_complete", total_candidates=len(results))
    return results[:5]


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    elif score >= 0.55:
        return "medium"
    return "low"


def _normalise_key(title: str, artist: str) -> str:
    """Normalise title+artist to a deduplication key."""
    t = title.lower().strip()
    a = artist.lower().strip()
    # Remove common suffixes
    for suffix in [" (official video)", " (official audio)", " (lyrics)", " (audio)"]:
        t = t.replace(suffix, "")
    return f"{a}|{t}"


def _fuzzy_match_key(title: str, candidates: dict) -> str | None:
    """
    Try to find an existing candidate that matches this title.
    Simple substring matching — good enough for most cases.
    """
    title_lower = title.lower()
    for key in candidates.keys():
        candidate_title = key.split("|", 1)[-1]
        if candidate_title and (
            candidate_title in title_lower or
            title_lower in candidate_title or
            _word_overlap(candidate_title, title_lower) > 0.6
        ):
            return key
    return None


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on words."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
