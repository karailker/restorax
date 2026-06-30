# RestoraX — Status

**Last updated:** 2026-06-30
**Branch:** `feat/arch-vendoring-benchmarks` (5 commits ahead of `main`, tree clean except docs)

Point-in-time snapshot of where we are. Roadmap lives in [`PLAN.md`](PLAN.md). Update this in the same session state changes.

---

## Where we left off

Deep code review of the 10 working ("real") restorers + research sprint for benchmark/promo assets. Two new tracks opened: **arch vendoring for the 15 remaining models**, and **industry-standard benchmark + promo assets**.

## What's done

- **All five plan tracks shipped to `main`:** SP1 (backend foundations), SP2 (DAG engine), SP3 (modern UI M1–M4 + CORS/mobile/event-loop fixes), SP4 (ComfyUI node pack, PR #23), real samples (25-model benchmark docs + slider).
- **YADIF deinterlacer merged** → 26 restorers total (PLAN.md §2 still says 25 — stale).
- **Arch-vendoring branch (unmerged, 5 commits):**
  - `8529d4f` — fix BasicVSR++ wrong arch class (`BasicVSR` → `BasicVSRPlusPlus`, num_block 30→7); dead HF repos → manual-download `RestorerLoadError`; GFPGAN repo → `nlightcho/gfpgan_v14`.
  - `238d912` — vendor VRT arch (`vrt_arch.py`, BSD) from JingyunLiang/VRT.
  - `648a68d` — vendor CodeFormer arch (`codeformer_arch.py` + `vqgan_arch.py`, MIT) from sczhou/CodeFormer.
  - `9957e1b` / `3d4e79d` — real GFPGAN/CodeFormer before/after face PNGs.

## In flight

- **Arch vendoring — 15 remaining models** (parallel sub-agent work, disjoint files). See `PLAN.md` §4.5.
- **Benchmark + promo assets** (Xiph / Netflix Chimera / DAVIS). See `PLAN.md` §4.6.

## Code review findings (10 "real" models)

7 parallel review forks ran; 2 returned with verified findings, 5 still pending. Verified so far:

- **HIGH (systemic, audio):** runtime silent-passthrough `except Exception: return audio.copy()` in `demucs.py` / `voicefixer.py` / `rnnoise.py`. Runtime failure → unchanged audio, job reports success. The load-time canary (`test_no_silent_stubs`) doesn't see this. Fix: raise a `RestorerProcessingError` (or re-raise after logging). Audio trio also has zero unit tests.
- **MEDIUM:** demucs silent empty-stem selection (all-zeros, no warning); demucs hardcoded 44100 (no resampling).
- **Security — CLEAN (verified):** yadif `subprocess.run` is list-form, no `shell=True`, literal ffmpeg args, tempdir-scoped, `timeout=120`. Audio temp files use `TemporaryDirectory` and clean up. No creds, no path traversal, no unsafe `torch.load` in verified files.

Pending: real_esrgan, codeformer, gfpgan, rife, stabilization (deep_flow_stab + gavs — manifest shows both `elapsed 0.0s`, suspicious for no-op; fork will confirm real work vs pass-through).

## Open gaps (unchanged from PLAN.md §5)

No Alembic migrations; test suite not isolated when run as a whole (15 pre-existing failures reproduce on `main` — env/teardown leakage, NOT mine); audio restorers not DAG-`restore`-runnable; `models.py` `object.__new__` fragility; `s3_secret_key` dev default; no frontend ESLint config.

## Env gotcha (this session)

Run tests with `PYTHONNOUSERSITE=1 /home/ilker/anaconda3/envs/restorax/bin/python -m pytest`. Without `PYTHONNOUSERSITE=1`, `~/.local` user-site `importlib_metadata` shadows the conda env and breaks `tqdm`/`torch` import at collection. (Captured in project `CLAUDE.md` §7.)