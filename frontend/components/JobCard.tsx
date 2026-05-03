"use client";

import Link from "next/link";
import { type Job } from "@/lib/api";
import ProgressBar from "./ProgressBar";

const STATUS_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  pending:   { bg: "bg-gray-100",   text: "text-gray-500",  dot: "bg-gray-400" },
  queued:    { bg: "bg-amber-50",   text: "text-amber-700", dot: "bg-amber-400 animate-pulse" },
  running:   { bg: "bg-indigo-50",  text: "text-indigo-700",dot: "bg-indigo-500 animate-pulse" },
  completed: { bg: "bg-green-50",   text: "text-green-700", dot: "bg-green-500" },
  failed:    { bg: "bg-red-50",     text: "text-red-700",   dot: "bg-red-500" },
  cancelled: { bg: "bg-gray-100",   text: "text-gray-500",  dot: "bg-gray-400" },
};

const PIPELINE_ICONS: Record<string, string> = {
  sr_x4:             "⬆",
  sr_x4_face:        "👤",
  classic_film:      "🎞",
  classic_film_audio:"🎞",
  anime_upscale:     "✨",
  vhs_restoration:   "📼",
  newsreel:          "📰",
};

interface Props {
  job: Job;
}

export default function JobCard({ job }: Props) {
  const filename = job.input_path.split("/").at(-1) ?? job.input_path;
  const style = STATUS_STYLES[job.status] ?? STATUS_STYLES.pending;
  const icon = PIPELINE_ICONS[job.pipeline_id] ?? "🎬";
  const elapsed = job.completed_at && job.started_at
    ? ((new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000).toFixed(1) + "s"
    : null;

  return (
    <Link href={`/jobs/${job.id}`} className="block group">
      <div className="bg-white rounded-xl border border-gray-100 hover:border-indigo-200
        hover:shadow-md transition-all p-4 space-y-3">

        {/* Top row */}
        <div className="flex items-start gap-3">
          <span className="text-2xl shrink-0 mt-0.5">{icon}</span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-gray-900 truncate">{filename}</p>
            <p className="text-xs text-gray-400 mt-0.5">
              {job.pipeline_id.replace(/_/g, " ")}
              {elapsed && <span className="ml-2 text-gray-300">· {elapsed}</span>}
            </p>
          </div>
          <span className={`shrink-0 inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full capitalize ${style.bg} ${style.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
            {job.status}
          </span>
        </div>

        <ProgressBar
          jobId={job.id}
          initialStatus={job.status}
          initialProgress={job.progress}
        />

        {/* Metrics row */}
        {job.metrics && Object.keys(job.metrics).length > 0 && (
          <div className="flex gap-4 text-xs text-gray-500 pt-1 border-t border-gray-50">
            {job.metrics.psnr !== undefined && (
              <span>PSNR <strong className="text-gray-700">{job.metrics.psnr.toFixed(1)}</strong> dB</span>
            )}
            {job.metrics.ssim !== undefined && (
              <span>SSIM <strong className="text-gray-700">{job.metrics.ssim.toFixed(3)}</strong></span>
            )}
            {job.metrics.lpips !== undefined && (
              <span>LPIPS <strong className="text-gray-700">{job.metrics.lpips.toFixed(3)}</strong></span>
            )}
          </div>
        )}

        {/* Error inline */}
        {job.error && (
          <p className="text-xs text-red-600 bg-red-50 rounded px-2 py-1 truncate">{job.error}</p>
        )}
      </div>
    </Link>
  );
}
