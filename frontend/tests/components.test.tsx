import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import JobCard from "../components/JobCard";
import ProgressBar from "../components/ProgressBar";
import JobForm from "../components/JobForm";
import type { Job } from "../lib/api";

// ── mock next/link so it renders a plain <a> ──────────────────────────────────
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode; [k: string]: unknown }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

// ── mock wsJobProgress to avoid real WebSocket ────────────────────────────────
vi.mock("../lib/api", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../lib/api")>();
  return {
    ...mod,
    wsJobProgress: () => {
      const ws = {
        onmessage: null as unknown,
        onerror: null as unknown,
        close: vi.fn(),
      };
      return ws as unknown as WebSocket;
    },
    submitJob: vi.fn(),
  };
});

// ── helpers ───────────────────────────────────────────────────────────────────

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "test-id",
    status: "queued",
    progress: 0.5,
    pipeline_id: "sr_x4",
    input_path: "/uploads/film.mp4",
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

// ── JobCard ───────────────────────────────────────────────────────────────────

describe("JobCard", () => {
  it("renders filename from input_path", () => {
    render(<JobCard job={makeJob()} />);
    expect(screen.getByText("film.mp4")).toBeInTheDocument();
  });

  it("renders pipeline id", () => {
    render(<JobCard job={makeJob()} />);
    // pipeline_id is displayed with underscores replaced by spaces
    expect(screen.getByText(/sr.x4/i)).toBeInTheDocument();
  });

  it("shows status badge", () => {
    render(<JobCard job={makeJob({ status: "completed" })} />);
    const matches = screen.getAllByText("completed");
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it("shows PSNR metric when present", () => {
    render(<JobCard job={makeJob({ metrics: { psnr: 28.4, ssim: 0.82 } })} />);
    expect(screen.getByText(/28\.4/)).toBeInTheDocument();
    expect(screen.getByText(/0\.820/)).toBeInTheDocument();
  });

  it("does not render metric section when metrics empty", () => {
    render(<JobCard job={makeJob({ metrics: {} })} />);
    expect(screen.queryByText(/PSNR/)).not.toBeInTheDocument();
  });

  it("links to job detail page", () => {
    render(<JobCard job={makeJob({ id: "abc" })} />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "/jobs/abc");
  });
});

// ── ProgressBar ───────────────────────────────────────────────────────────────

describe("ProgressBar", () => {
  it("displays initial progress percentage", () => {
    render(<ProgressBar jobId="j1" initialStatus="running" initialProgress={0.42} />);
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("displays initial status text", () => {
    render(<ProgressBar jobId="j1" initialStatus="queued" initialProgress={0} />);
    expect(screen.getByText("queued")).toBeInTheDocument();
  });

  it("shows 100% for completed job", () => {
    render(<ProgressBar jobId="j1" initialStatus="completed" initialProgress={1.0} />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("does not crash for terminal status", () => {
    render(<ProgressBar jobId="j1" initialStatus="completed" initialProgress={1} />);
    expect(screen.getByText("completed")).toBeInTheDocument();
  });
});

// ── JobForm ───────────────────────────────────────────────────────────────────

describe("JobForm", () => {
  afterEach(() => { vi.clearAllMocks(); });

  it("renders file input and submit button", () => {
    render(<JobForm onJobCreated={vi.fn()} />);
    expect(screen.getByRole("button", { name: /restore/i })).toBeInTheDocument();
  });

  it("submit is disabled/inert without a file", async () => {
    const cb = vi.fn();
    render(<JobForm onJobCreated={cb} />);
    fireEvent.submit(screen.getByRole("button", { name: /restore/i }).closest("form")!);
    expect(cb).not.toHaveBeenCalled();
  });

  it("calls submitJob and invokes onJobCreated on success", async () => {
    const { submitJob } = await import("../lib/api");
    const job = makeJob({ status: "queued" });
    vi.mocked(submitJob).mockResolvedValueOnce(job);

    const cb = vi.fn();
    render(<JobForm onJobCreated={cb} />);

    const fileInput = screen.getByRole("button", { name: /restore/i })
      .closest("form")!
      .querySelector("input[type=file]")!;

    const file = new File(["data"], "clip.mp4", { type: "video/mp4" });
    await userEvent.upload(fileInput as HTMLInputElement, file);

    fireEvent.submit(fileInput.closest("form")!);

    await waitFor(() => expect(cb).toHaveBeenCalledWith(job));
  });

  it("shows error message on submit failure", async () => {
    const { submitJob } = await import("../lib/api");
    vi.mocked(submitJob).mockRejectedValueOnce(new Error("Server error"));

    render(<JobForm onJobCreated={vi.fn()} />);
    const form = screen.getByRole("button", { name: /restore/i }).closest("form")!;
    const fileInput = form.querySelector("input[type=file]")!;

    const file = new File(["x"], "v.mp4", { type: "video/mp4" });
    await userEvent.upload(fileInput as HTMLInputElement, file);
    fireEvent.submit(form);

    await waitFor(() => expect(screen.getByText(/server error/i)).toBeInTheDocument());
  });
});
