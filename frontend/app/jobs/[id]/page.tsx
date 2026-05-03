"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import ProgressBar from "@/components/ProgressBar";
import { fetchJob, type Job } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const PIPELINE_LABELS: Record<string, string> = {
  sr_x4: "4× Super-Resolution",
  sr_x4_face: "4× SR + Face Restoration",
  classic_film: "Classic Film Pipeline",
  classic_film_audio: "Classic Film + Audio",
  anime_upscale: "Anime / Illustration 2×",
  vhs_restoration: "VHS Tape Restoration",
  newsreel: "Newsreel / Archival",
};

const STATUS_COLORS: Record<string, string> = {
  pending: "text-gray-500",
  queued: "text-amber-600",
  running: "text-indigo-600",
  completed: "text-green-600",
  failed: "text-red-600",
  cancelled: "text-gray-400",
};

export default function JobDetailPage() {
  const { id: jobId } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJob(jobId).then(setJob).catch((e) => setError(e.message));
  }, [jobId]);

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow p-8 text-center space-y-3 max-w-sm w-full">
          <span className="text-3xl">⚠️</span>
          <p className="text-red-600 font-medium">{error}</p>
          <Link href="/" className="block text-sm text-indigo-600 hover:underline">← Back to jobs</Link>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-2 text-gray-400">
          <span className="animate-spin inline-block w-6 h-6 border-2 border-gray-300 border-t-indigo-500 rounded-full" />
          <p className="text-sm">Loading job…</p>
        </div>
      </div>
    );
  }

  const filename = job.input_path.split("/").at(-1) ?? job.input_path;
  const pipelineLabel = PIPELINE_LABELS[job.pipeline_id] ?? job.pipeline_id;
  const statusColor = STATUS_COLORS[job.status] ?? "text-gray-500";
  const duration = job.completed_at && job.started_at
    ? ((new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000).toFixed(1)
    : null;

  const downloadUrl = `${API_BASE}/jobs/${job.id}/download`;

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-10 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center gap-2 text-sm">
          <Link href="/" className="text-indigo-600 hover:text-indigo-800 font-medium">← Jobs</Link>
          <span className="text-gray-300">/</span>
          <span className="text-gray-700 truncate max-w-xs">{filename}</span>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 py-8 space-y-5">
        {/* Status card */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-lg font-semibold text-gray-900 truncate">{filename}</h1>
              <p className="text-sm text-gray-500 mt-0.5">{pipelineLabel}</p>
            </div>
            <span className={`text-sm font-semibold capitalize shrink-0 ${statusColor}`}>
              {job.status}
            </span>
          </div>

          <ProgressBar
            jobId={job.id}
            initialStatus={job.status}
            initialProgress={job.progress}
            onComplete={(path) =>
              setJob((j) => j ? { ...j, status: "completed", output_path: path, progress: 1.0 } : j)
            }
            onFail={(err) =>
              setJob((j) => j ? { ...j, status: "failed", error: err } : j)
            }
          />

          {/* Timestamps */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs text-gray-500">
            <div>
              <p className="font-medium text-gray-400 uppercase tracking-wide">Created</p>
              <p>{new Date(job.created_at).toLocaleString()}</p>
            </div>
            {job.started_at && (
              <div>
                <p className="font-medium text-gray-400 uppercase tracking-wide">Started</p>
                <p>{new Date(job.started_at).toLocaleString()}</p>
              </div>
            )}
            {duration && (
              <div>
                <p className="font-medium text-gray-400 uppercase tracking-wide">Duration</p>
                <p className="text-green-600 font-medium">{duration}s</p>
              </div>
            )}
          </div>

          {job.error && (
            <div className="rounded-lg bg-red-50 border border-red-100 px-4 py-3">
              <p className="text-sm font-medium text-red-700">Error</p>
              <p className="text-xs text-red-600 mt-0.5">{job.error}</p>
            </div>
          )}
        </div>

        {/* Quality Metrics */}
        {job.metrics && Object.keys(job.metrics).length > 0 && (
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Quality Metrics</h2>
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(job.metrics).map(([key, val]) => (
                <div key={key} className="text-center p-3 bg-indigo-50 rounded-xl">
                  <p className="text-xs text-indigo-500 uppercase tracking-wide font-medium">{key}</p>
                  <p className="text-2xl font-bold text-indigo-700 mt-1">
                    {typeof val === "number" ? val.toFixed(key === "psnr" ? 1 : 3) : val}
                  </p>
                  {key === "psnr" && <p className="text-xs text-indigo-400">dB</p>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Download / Result */}
        {job.status === "completed" && job.output_path && (
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-gray-700">Restoration Complete</h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  Output: {job.output_path.split("/").at(-1)}
                </p>
              </div>
              <span className="text-2xl">✅</span>
            </div>

            <a
              href={downloadUrl}
              download
              className="inline-flex items-center justify-center gap-2 w-full sm:w-auto
                px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold
                rounded-xl transition-colors shadow-sm"
            >
              ↓ Download Restored Video
            </a>

            <p className="text-xs text-gray-400">
              Tip: Open the video in VLC or compare with the original using a video editor.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
