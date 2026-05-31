"use client";

import { useState, useRef } from "react";
import { Search, Music, Upload, Link, Loader2, ExternalLink, AlertCircle } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Platform = {
  spotify?: string | null;
  youtube?: string | null;
  soundcloud?: string | null;
  apple_music?: string | null;
};

type EditInfo = {
  detected: boolean;
  edit_type: string;
  speed_factor: number;
  pitch_shift_semitones: number;
};

type Result = {
  rank: number;
  title: string;
  artist: string;
  album: string;
  release_year: string;
  confidence: number;
  confidence_label: string;
  platforms: Platform;
  edit_info: EditInfo;
  artwork_url: string;
  preview_url: string;
};

type AudioFeatures = {
  bpm: number;
  key: string;
  mode: string;
  estimated_speed_factor: number;
};

type JobResult = {
  status: string;
  match_type: string;
  results: Result[];
  audio_features: AudioFeatures | null;
  processing_time_ms: number;
  error: string | null;
};

export default function Home() {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [error, setError] = useState("");
  const [statusMsg, setStatusMsg] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const poll = async (jobId: string) => {
    const maxAttempts = 30;
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const res = await fetch(`${API_URL}/api/v1/results/${jobId}`);
      const data = await res.json();
      if (data.status === "complete" || data.status === "failed") {
        return data;
      }
      setStatusMsg(i < 3 ? "Downloading audio..." : i < 8 ? "Analysing sound..." : "Searching platforms...");
    }
    throw new Error("Timed out waiting for results");
  };

  const handleSubmit = async () => {
    if (!url && !file) return;
    setLoading(true);
    setJobResult(null);
    setError("");
    setStatusMsg("Submitting...");

    try {
      const formData = new FormData();
      if (file) formData.append("file", file);
      else formData.append("url", url);

      const res = await fetch(`${API_URL}/api/v1/identify`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to submit");
      }

      const { job_id } = await res.json();
      setStatusMsg("Processing...");
      const result = await poll(job_id);

      if (result.status === "failed") {
        throw new Error(result.error || "Processing failed");
      }

      setJobResult(result);
    } catch (e: any) {
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  };

  const confidenceColor = (label: string) => {
    if (label === "high") return "text-emerald-400 bg-emerald-400/10";
    if (label === "medium") return "text-amber-400 bg-amber-400/10";
    return "text-red-400 bg-red-400/10";
  };

  const platformLinks = (platforms: Platform) => {
    const links = [];
    if (platforms.spotify) links.push({ name: "Spotify", url: platforms.spotify, color: "bg-green-500 hover:bg-green-400" });
    if (platforms.youtube) links.push({ name: "YouTube", url: platforms.youtube, color: "bg-red-500 hover:bg-red-400" });
    if (platforms.soundcloud) links.push({ name: "SoundCloud", url: platforms.soundcloud, color: "bg-orange-500 hover:bg-orange-400" });
    if (platforms.apple_music) links.push({ name: "Apple Music", url: platforms.apple_music, color: "bg-pink-500 hover:bg-pink-400" });
    return links;
  };

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <div className="border-b border-white/5 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-violet-500 flex items-center justify-center">
          <Music className="w-4 h-4 text-white" />
        </div>
        <span className="font-semibold text-lg tracking-tight">SoundMatch</span>
        <span className="text-xs text-white/30 ml-1">beta</span>
      </div>

      <div className="max-w-2xl mx-auto px-6 py-16">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight mb-3">
            Find any song.<br />
            <span className="text-violet-400">Even the niche ones.</span>
          </h1>
          <p className="text-white/40 text-lg">
            Paste a TikTok, Instagram or YouTube link — we'll identify the song
            and find it across Spotify, YouTube and SoundCloud.
          </p>
        </div>

        {/* Input card */}
        <div className="bg-white/[0.03] border border-white/8 rounded-2xl p-6 mb-6">
          {/* URL input */}
          <div className="flex gap-3 mb-4">
            <div className="flex-1 flex items-center gap-3 bg-white/5 border border-white/8 rounded-xl px-4 py-3">
              <Link className="w-4 h-4 text-white/30 flex-shrink-0" />
              <input
                type="text"
                value={url}
                onChange={(e) => { setUrl(e.target.value); setFile(null); }}
                placeholder="Paste a TikTok, Instagram or YouTube URL..."
                className="flex-1 bg-transparent text-sm outline-none placeholder:text-white/25"
                disabled={loading}
              />
            </div>
          </div>

          {/* Divider */}
          <div className="flex items-center gap-3 mb-4">
            <div className="flex-1 h-px bg-white/5" />
            <span className="text-xs text-white/20">or upload a file</span>
            <div className="flex-1 h-px bg-white/5" />
          </div>

          {/* File upload */}
          <div
            onClick={() => !loading && fileInputRef.current?.click()}
            className={`border border-dashed border-white/10 rounded-xl px-4 py-4 flex items-center gap-3 cursor-pointer hover:border-violet-500/40 hover:bg-violet-500/5 transition-all mb-5 ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <Upload className="w-4 h-4 text-white/30" />
            <span className="text-sm text-white/30">
              {file ? file.name : "MP3, MP4, WAV, M4A up to 50MB"}
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp3,.mp4,.wav,.m4a,.ogg,.webm,.aac,.flac"
              className="hidden"
              onChange={(e) => { setFile(e.target.files?.[0] || null); setUrl(""); }}
            />
          </div>

          {/* Submit button */}
          <button
            onClick={handleSubmit}
            disabled={loading || (!url && !file)}
            className="w-full bg-violet-500 hover:bg-violet-400 disabled:opacity-30 disabled:cursor-not-allowed text-white font-medium py-3 rounded-xl flex items-center justify-center gap-2 transition-all"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>{statusMsg || "Processing..."}</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                <span>Identify Song</span>
              </>
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 mb-6">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span className="text-sm text-red-300">{error}</span>
          </div>
        )}

        {/* Results */}
        {jobResult && (
          <div className="space-y-4">
            {/* Summary bar */}
            <div className="flex items-center justify-between text-xs text-white/30 px-1">
              <span>
                {jobResult.results.length > 0
                  ? `${jobResult.results.length} result${jobResult.results.length > 1 ? "s" : ""} found`
                  : "No matches found"}
                {jobResult.match_type === "edit_detected" && " · edit detected"}
              </span>
              <span>{(jobResult.processing_time_ms / 1000).toFixed(1)}s</span>
            </div>

            {/* No match */}
            {jobResult.results.length === 0 && (
              <div className="bg-white/[0.03] border border-white/8 rounded-2xl p-8 text-center">
                <Music className="w-10 h-10 text-white/10 mx-auto mb-3" />
                <p className="text-white/40 text-sm">
                  Couldn't identify this sound. It may be an original recording
                  not available on any platform yet.
                </p>
                {jobResult.audio_features && (
                  <div className="flex justify-center gap-4 mt-4 text-xs text-white/20">
                    <span>BPM {jobResult.audio_features.bpm}</span>
                    <span>{jobResult.audio_features.key} {jobResult.audio_features.mode}</span>
                  </div>
                )}
              </div>
            )}

            {/* Result cards */}
            {jobResult.results.map((result, i) => (
              <div key={i} className="bg-white/[0.03] border border-white/8 rounded-2xl p-5">
                <div className="flex items-start gap-4">
                  {/* Artwork */}
                  <div className="w-14 h-14 rounded-lg bg-white/5 flex-shrink-0 overflow-hidden">
                    {result.artwork_url ? (
                      <img src={result.artwork_url} alt={result.title} className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Music className="w-5 h-5 text-white/20" />
                      </div>
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <div>
                        <h3 className="font-semibold text-sm truncate">{result.title || "Unknown title"}</h3>
                        <p className="text-white/40 text-xs truncate">{result.artist || "Unknown artist"}</p>
                      </div>
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0 ${confidenceColor(result.confidence_label)}`}>
                        {Math.round(result.confidence * 100)}%
                      </span>
                    </div>

                    {/* Edit badge */}
                    {result.edit_info.detected && (
                      <div className="text-xs text-amber-400/70 mb-2">
                        ⚡ {result.edit_info.edit_type.replace("_", " ")} · {result.edit_info.speed_factor}x speed
                      </div>
                    )}

                    {/* Platform links */}
                    <div className="flex flex-wrap gap-2 mt-2">
                      {platformLinks(result.platforms).map((link) => (
                        <a
                          key={link.name}
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={`${link.color} text-white text-xs font-medium px-3 py-1 rounded-full flex items-center gap-1 transition-colors`}
                        >
                          {link.name}
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Audio features */}
            {jobResult.audio_features && jobResult.results.length > 0 && (
              <div className="flex gap-4 px-1 text-xs text-white/20">
                <span>BPM {jobResult.audio_features.bpm}</span>
                <span>{jobResult.audio_features.key} {jobResult.audio_features.mode}</span>
                {jobResult.audio_features.estimated_speed_factor !== 1.0 && (
                  <span>Speed {jobResult.audio_features.estimated_speed_factor}x</span>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}