import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { fetchJobs, submitJob } from "@/lib/api";
import type { Job } from "@/types";
import { StatsStrip } from "@/components/dashboard/StatsStrip";
import { JobsTable } from "@/components/dashboard/JobsTable";
import { PresetCard } from "@/components/dashboard/PresetCard";
import { PRESETS, type Preset } from "@/components/dashboard/presets";

const POLL_INTERVAL_MS = 5000;

/**
 * Dashboard — home view: job stats, quick-launch presets, and recent jobs.
 */
export default function Dashboard() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploadingId, setUploadingId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pendingPreset = useRef<Preset | null>(null);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const data = await fetchJobs();
        if (!active) return;
        setJobs(data);
        setError(null);
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setLoading(false);
      }
    };

    void load();
    const interval = setInterval(() => void load(), POLL_INTERVAL_MS);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const handlePresetClick = (preset: Preset) => {
    if (uploadingId) return;
    pendingPreset.current = preset;
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    const preset = pendingPreset.current;
    e.target.value = ""; // allow re-selecting the same file
    pendingPreset.current = null;
    if (!file || !preset) return;

    setUploadingId(preset.pipelineId);
    try {
      const job = await submitJob(file, { pipelineId: preset.pipelineId });
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setUploadingId(null);
    }
  };

  return (
    <div className="mx-auto max-w-5xl p-8">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Launch a restoration or track recent jobs
          </p>
        </div>
        <Button asChild>
          <Link to="/builder">
            <Plus />
            New pipeline
          </Link>
        </Button>
      </header>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {error && (
        <Card className="mb-6 border-[var(--color-destructive)]/40">
          <CardContent className="pt-5 text-sm text-[var(--color-destructive)]">
            Something went wrong: {error}
          </CardContent>
        </Card>
      )}

      <div className="mb-8">
        <StatsStrip jobs={jobs} />
      </div>

      <section className="mb-8">
        <h2 className="mb-3 text-sm font-medium text-[var(--color-muted-foreground)]">
          Quick launch
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {PRESETS.map((preset) => (
            <PresetCard
              key={preset.pipelineId}
              preset={preset}
              onClick={handlePresetClick}
              uploading={uploadingId === preset.pipelineId}
              disabled={uploadingId !== null}
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium text-[var(--color-muted-foreground)]">
          Recent jobs
        </h2>
        {loading && jobs.length === 0 && !error ? (
          <Card>
            <CardContent className="py-12 text-center text-sm text-[var(--color-muted-foreground)]">
              Loading jobs…
            </CardContent>
          </Card>
        ) : (
          <JobsTable jobs={jobs} />
        )}
      </section>
    </div>
  );
}
