import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  fetchJobs,
  fetchJob,
  submitJob,
  fetchModels,
  wsJobProgress,
  type Job,
} from "../lib/api";

// ── helpers ───────────────────────────────────────────────────────────────────

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "abc123",
    status: "queued",
    progress: 0,
    pipeline_id: "sr_x4",
    input_path: "/data/film.mp4",
    output_path: null,
    error: null,
    metrics: {},
    created_at: "2026-04-25T12:00:00Z",
    started_at: null,
    completed_at: null,
    celery_task_id: null,
    ...overrides,
  };
}

function mockFetch(body: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
  });
}

// ── fetchJobs ─────────────────────────────────────────────────────────────────

describe("fetchJobs", () => {
  beforeEach(() => {
    global.fetch = mockFetch({ jobs: [makeJob()] });
  });
  afterEach(() => { vi.restoreAllMocks(); });

  it("calls /jobs with limit param", async () => {
    await fetchJobs(5);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("limit=5"),
      expect.any(Object),
    );
  });

  it("returns array of jobs", async () => {
    const jobs = await fetchJobs();
    expect(jobs).toHaveLength(1);
    expect(jobs[0].id).toBe("abc123");
  });

  it("throws on non-ok response", async () => {
    global.fetch = mockFetch({ detail: "not found" }, 404);
    await expect(fetchJobs()).rejects.toThrow("404");
  });
});

// ── fetchJob ──────────────────────────────────────────────────────────────────

describe("fetchJob", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it("returns job by id", async () => {
    global.fetch = mockFetch(makeJob({ id: "xyz" }));
    const job = await fetchJob("xyz");
    expect(job.id).toBe("xyz");
  });

  it("throws on 404", async () => {
    global.fetch = mockFetch({}, 404);
    await expect(fetchJob("missing")).rejects.toThrow("missing");
  });
});

// ── submitJob ─────────────────────────────────────────────────────────────────

describe("submitJob", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it("POSTs multipart form with file and pipeline_id", async () => {
    global.fetch = mockFetch(makeJob({ status: "queued" }));
    const file = new File(["data"], "film.mp4", { type: "video/mp4" });
    const job = await submitJob(file, "sr_x4");
    expect(job.status).toBe("queued");
    const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toContain("/jobs");
    expect(init.method).toBe("POST");
    const body = init.body as FormData;
    expect(body.get("pipeline_id")).toBe("sr_x4");
    expect(body.get("file")).toBeInstanceOf(File);
  });

  it("throws with server error detail", async () => {
    global.fetch = mockFetch({ detail: "Unsupported format" }, 422);
    const file = new File(["x"], "bad.avi");
    await expect(submitJob(file, "sr_x4")).rejects.toThrow("Unsupported format");
  });

  it("passes optional output options", async () => {
    global.fetch = mockFetch(makeJob());
    const file = new File(["d"], "v.mp4");
    await submitJob(file, "sr_x4", { outputCodec: "libx265", outputCrf: 28, preserveAudio: false });
    const body = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1].body as FormData;
    expect(body.get("output_codec")).toBe("libx265");
    expect(body.get("output_crf")).toBe("28");
    expect(body.get("preserve_audio")).toBe("false");
  });
});

// ── fetchModels ───────────────────────────────────────────────────────────────

describe("fetchModels", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it("returns restorer list", async () => {
    global.fetch = mockFetch({ restorers: [{ name: "real_esrgan_x4plus", loaded: false }] });
    const models = await fetchModels();
    expect(models[0].name).toBe("real_esrgan_x4plus");
  });

  it("throws on error", async () => {
    global.fetch = mockFetch({}, 500);
    await expect(fetchModels()).rejects.toThrow("Failed to fetch models");
  });
});

// ── wsJobProgress ─────────────────────────────────────────────────────────────

describe("wsJobProgress", () => {
  it("returns WebSocket pointing at ws:// url", () => {
    const MockWS = vi.fn();
    vi.stubGlobal("WebSocket", MockWS);
    wsJobProgress("job-42");
    const url: string = MockWS.mock.calls[0][0];
    expect(url).toMatch(/^ws/);
    expect(url).toContain("job-42");
    vi.unstubAllGlobals();
  });
});
