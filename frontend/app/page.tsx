"use client";

import { useCallback, useEffect, useState } from "react";
import JobCard from "@/components/JobCard";
import JobForm from "@/components/JobForm";
import { fetchJobs, type Job } from "@/lib/api";

export default function HomePage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");

  const loadJobs = useCallback(async () => {
    try {
      const data = await fetchJobs(50);
      setJobs(data);
    } catch {
      // API not reachable — show empty state
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadJobs();
    const id = setInterval(loadJobs, 5000); // auto-refresh every 5 s
    return () => clearInterval(id);
  }, [loadJobs]);

  function handleJobCreated(job: Job) {
    setJobs((prev) => [job, ...prev]);
  }

  const statusCounts = jobs.reduce<Record<string, number>>((acc, j) => {
    acc[j.status] = (acc[j.status] ?? 0) + 1;
    return acc;
  }, {});

  const filtered = filter === "all" ? jobs : jobs.filter((j) => j.status === filter);

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Nav */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">🎬</span>
            <span className="font-bold text-gray-900">RestoraX</span>
            <span className="hidden sm:block text-xs bg-indigo-100 text-indigo-700 font-medium px-2 py-0.5 rounded-full">
              AI Video Restoration
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400">
            {statusCounts.running ? (
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                {statusCounts.running} running
              </span>
            ) : null}
            <button
              onClick={loadJobs}
              className="text-indigo-600 hover:text-indigo-800 font-medium transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left: form */}
          <div className="lg:col-span-1">
            <JobForm onJobCreated={handleJobCreated} />

            {/* Stats panel */}
            {jobs.length > 0 && (
              <div className="mt-4 bg-white rounded-2xl border border-gray-100 p-4">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Session</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { label: "Total", value: jobs.length, color: "text-gray-700" },
                    { label: "Done", value: statusCounts.completed ?? 0, color: "text-green-600" },
                    { label: "Running", value: statusCounts.running ?? 0, color: "text-indigo-600" },
                    { label: "Failed", value: statusCounts.failed ?? 0, color: "text-red-600" },
                  ].map((s) => (
                    <div key={s.label} className="text-center p-2 rounded-lg bg-gray-50">
                      <p className={`text-xl font-bold ${s.color}`}>{s.value}</p>
                      <p className="text-xs text-gray-400">{s.label}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right: job list */}
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-gray-800">
                Jobs
                {filtered.length > 0 && (
                  <span className="ml-2 text-xs font-normal text-gray-400">{filtered.length}</span>
                )}
              </h2>

              {/* Filter tabs */}
              <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5">
                {["all", "running", "completed", "failed"].map((s) => (
                  <button
                    key={s}
                    onClick={() => setFilter(s)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors capitalize ${
                      filter === s
                        ? "bg-white text-gray-800 shadow-sm"
                        : "text-gray-500 hover:text-gray-700"
                    }`}
                  >
                    {s === "all" ? `All (${jobs.length})` : `${s} (${statusCounts[s] ?? 0})`}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div className="text-sm text-gray-400 text-center py-12">Loading…</div>
            ) : filtered.length === 0 ? (
              <div className="text-center py-16 bg-white rounded-2xl border border-gray-100">
                <span className="text-4xl">🎞️</span>
                <p className="mt-3 text-sm font-medium text-gray-600">
                  {filter === "all" ? "No jobs yet" : `No ${filter} jobs`}
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  {filter === "all" ? "Upload a video to get started" : "Try a different filter"}
                </p>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2">
                {filtered.map((job) => <JobCard key={job.id} job={job} />)}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
