"use client";

import { useEffect, useState } from "react";
import { fetchModels, submitJob, type Job, type RestorerInfo } from "@/lib/api";

// Static pipeline catalog — matches configs/presets/*.yaml
const PIPELINES = [
  {
    id: "sr_x4",
    label: "4× Super-Resolution",
    description: "Real-ESRGAN blind upscaling — best all-round starting point",
    tags: ["SR", "4×"],
    category: "super_resolution",
  },
  {
    id: "sr_x4_face",
    label: "4× SR + Face Restoration",
    description: "Real-ESRGAN followed by CodeFormer — ideal for footage with faces",
    tags: ["SR", "4×", "Face"],
    category: "super_resolution",
  },
  {
    id: "classic_film",
    label: "Classic Film Pipeline",
    description: "Full restoration: deinterlace → SR → colorize → face enhance → frame interpolation",
    tags: ["SR", "Colorize", "Face", "Interpolate"],
    category: "classic",
  },
  {
    id: "classic_film_audio",
    label: "Classic Film + Audio",
    description: "Classic film pipeline with Demucs audio denoising",
    tags: ["SR", "Colorize", "Face", "Audio"],
    category: "classic",
  },
  {
    id: "anime_upscale",
    label: "Anime / Illustration 2×",
    description: "Waifu2x sharpened upscaling tuned for anime and line-art",
    tags: ["SR", "2×", "Anime"],
    category: "super_resolution",
  },
  {
    id: "vhs_restoration",
    label: "VHS Tape Restoration",
    description: "Deinterlace + denoise + SR — designed for VHS and Betamax recordings",
    tags: ["VHS", "Deinterlace", "SR"],
    category: "classic",
  },
  {
    id: "newsreel",
    label: "Newsreel / Archival",
    description: "Scratch removal + SR + colorization — for B&W archival footage",
    tags: ["Scratch", "SR", "Colorize"],
    category: "classic",
  },
] as const;

const CATEGORY_LABELS: Record<string, string> = {
  super_resolution: "Super-Resolution",
  classic: "Complete Pipelines",
};

interface Props {
  onJobCreated: (job: Job) => void;
}

export default function JobForm({ onJobCreated }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [pipeline, setPipeline] = useState<string>(PIPELINES[0].id);
  const [restoreAudio, setRestoreAudio] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<RestorerInfo[]>([]);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    fetchModels().then(setModels).catch(() => {});
  }, []);

  const selectedPipeline = PIPELINES.find((p) => p.id === pipeline) ?? PIPELINES[0];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const job = await submitJob(file, pipeline, {
        preserveAudio: restoreAudio,
      });
      onJobCreated(job);
      setFile(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type.startsWith("video/")) setFile(dropped);
  }

  const grouped = PIPELINES.reduce<Record<string, typeof PIPELINES[number][]>>((acc, p) => {
    (acc[p.category] ??= []).push(p);
    return acc;
  }, {});

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-4">
        <h2 className="text-lg font-semibold text-white">New Restoration Job</h2>
        <p className="text-indigo-200 text-xs mt-0.5">
          {models.length > 0 ? `${models.length} AI models available` : "Loading models…"}
        </p>
      </div>

      <div className="p-6 space-y-5">
        {/* Drop zone */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">Video file</label>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`relative rounded-xl border-2 border-dashed transition-colors ${
              dragOver ? "border-indigo-400 bg-indigo-50" :
              file ? "border-green-400 bg-green-50" : "border-gray-200 bg-gray-50"
            }`}
          >
            <input
              type="file"
              accept="video/*"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              required
            />
            <div className="py-8 text-center pointer-events-none">
              {file ? (
                <>
                  <span className="text-2xl">🎬</span>
                  <p className="mt-1 text-sm font-medium text-green-700">{file.name}</p>
                  <p className="text-xs text-green-600">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                </>
              ) : (
                <>
                  <span className="text-3xl">📂</span>
                  <p className="mt-1 text-sm font-medium text-gray-600">Drop video here or click to browse</p>
                  <p className="text-xs text-gray-400">MP4, MKV, AVI, MOV, WebM</p>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Pipeline picker */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">Pipeline</label>
          <select
            value={pipeline}
            onChange={(e) => setPipeline(e.target.value)}
            className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm bg-white
              focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent"
          >
            {Object.entries(grouped).map(([cat, pipes]) => (
              <optgroup key={cat} label={CATEGORY_LABELS[cat] ?? cat}>
                {pipes.map((p) => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </optgroup>
            ))}
          </select>

          {/* Pipeline info card */}
          <div className="mt-2 rounded-lg bg-indigo-50 border border-indigo-100 px-3 py-2">
            <p className="text-xs text-indigo-700">{selectedPipeline.description}</p>
            <div className="flex flex-wrap gap-1 mt-1.5">
              {selectedPipeline.tags.map((t) => (
                <span key={t} className="inline-block text-[10px] font-medium bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded">
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Audio restoration toggle */}
        <label className="flex items-center gap-3 cursor-pointer select-none">
          <button
            type="button"
            role="switch"
            aria-checked={restoreAudio}
            onClick={() => setRestoreAudio((v) => !v)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
              restoreAudio ? "bg-indigo-500" : "bg-gray-200"
            }`}
          >
            <span className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
              restoreAudio ? "translate-x-4" : "translate-x-1"
            }`} />
          </button>
          <div>
            <span className="text-sm font-medium text-gray-700">Audio restoration</span>
            <span className="block text-xs text-gray-400">Demucs noise removal + VoiceFixer speech enhancement</span>
          </div>
        </label>

        {error && (
          <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
        )}

        <button
          type="submit"
          disabled={loading || !file}
          className="w-full py-2.5 px-4 rounded-xl font-semibold text-white
            bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800
            disabled:opacity-40 disabled:cursor-not-allowed
            transition-colors shadow-sm"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
              Uploading…
            </span>
          ) : "Restore Video"}
        </button>
      </div>
    </form>
  );
}
