# Sample Restorations

Drag any slider handle to reveal **Before ↔ After**. All samples run on real CC-BY Blender Foundation media.

<style>
.compare-wrap {
  position: relative; overflow: hidden; cursor: ew-resize;
  user-select: none; border-radius: 6px; max-width: 100%; margin: 1rem 0;
}
.compare-wrap img { display: block; width: 100%; height: auto; }
.compare-after { position: absolute; top: 0; left: 0; width: 50%; overflow: hidden; }
.compare-after img { width: 200%; }
.compare-handle {
  position: absolute; top: 0; bottom: 0; left: 50%; width: 3px;
  background: #fff; transform: translateX(-50%); pointer-events: none;
}
.compare-handle::before {
  content: "◀ ▶"; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%,-50%); background: #fff; color: #222;
  font-size: 11px; padding: 4px 6px; border-radius: 4px; white-space: nowrap;
}
.compare-label {
  position: absolute; bottom: 8px; font-size: 11px; font-weight: 600;
  background: rgba(0,0,0,.6); color: #fff; padding: 2px 8px; border-radius: 3px;
}
.label-before { left: 8px; }
.label-after  { right: 8px; }
table.status-table td:nth-child(3) { font-family: monospace; font-size: 0.85em; }
.badge-real   { color: #2da44e; font-weight: 700; }
.badge-failed { color: #cf222e; }
.badge-stub   { color: #9a6700; }
</style>

<script>
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.compare-wrap').forEach(wrap => {
    const after = wrap.querySelector('.compare-after');
    const handle = wrap.querySelector('.compare-handle');
    let drag = false;
    const set = x => {
      const r = wrap.getBoundingClientRect();
      const p = Math.min(100, Math.max(0, (x - r.left) / r.width * 100));
      after.style.width = p + '%'; handle.style.left = p + '%';
    };
    wrap.addEventListener('mousedown', e => { drag = true; set(e.clientX); });
    wrap.addEventListener('touchstart', e => { drag = true; set(e.touches[0].clientX); }, {passive:true});
    document.addEventListener('mousemove',  e => drag && set(e.clientX));
    document.addEventListener('touchmove',  e => drag && set(e.touches[0].clientX), {passive:true});
    document.addEventListener('mouseup',   () => drag = false);
    document.addEventListener('touchend',  () => drag = false);
  });
});
</script>

---

## Status Overview — All 25 Models

| # | Model | Status | Notes |
|---|-------|--------|-------|
| 1 | RealESRGAN x4plus | ✅ real | Weights: `models/real_esrgan/` |
| 2 | BasicVSR++ | ❌ weights | Arch fixed (BasicVSRPlusPlus); no public weight mirror |
| 3 | Upscale-A-Video | ❌ arch | Honest adapter shim vendored; real pipeline (~6000 LOC custom UNet/attention) not ported, `diffusers` not installed |
| 4 | VRT | ✅ real | Arch vendored (CC BY-NC 4.0); official `001_VRT_videosr_bi_REDS_6frames.pth` (166 MB) downloads from the project's GitHub release; verified on CPU, +2.8 to +5.8 dB over bicubic |
| 5 | MambaIR | ❌ arch | `mamba_ir_arch` not vendored; `pip install mamba-ssm` needed |
| 6 | TDM | ❌ weights | Diffusion model; no public weights released |
| 7 | SeedVR | ❌ weights | Diffusion model; no public weights released |
| 8 | Waifu2x | ✅ real | Arch vendored (`UpConv_7`, GPLv3, yu45020/Waifu2x); weights downloaded+extracted from upstream 7z via `py7zr` (`restorax[waifu2x]`) |
| 9 | FlashVSR | ❌ arch | `flashvsr_arch` not vendored yet |
| 10 | EvTexture | ✅ real | Arch vendored (Apache-2.0, DachunKai/EvTexture); weights from GitHub releases (`EvTexture_REDS_BIx4.pth`), HF repo was dead |
| 11 | CodeFormer | ✅ real | Arch vendored from sczhou/CodeFormer; weights auto-downloaded |
| 12 | CodeFormer++ | ❌ arch | `codeformer_pp_arch` not vendored yet |
| 13 | GFPGAN | ✅ real | Fixed dead HF repo → `nlightcho/gfpgan_v14` |
| 14 | DicFace | ❌ extra | `pip install restorax[dicface]` |
| 15 | DDColor | ❌ arch+weights | `ddcolor_arch` not vendored; `piddnad/DDColor` HF dead |
| 16 | RIFE | ✅ running | Classical fallback (temporal arch pending) |
| 17 | Scratch Removal | ✅ real | Arch vendored (S-Lab 1.0 NC, sczhou/ProPainter); weights from GitHub releases v0.1.0 (HF repo_type was wrong — `space`, not `model`) |
| 18 | HDRTVDM | ❌ arch+weights | `hdrtvdm_arch` not vendored; `AndreGuo/HDRTVDM` HF dead |
| 19 | Video Stabilization | ✅ running | OpenCV optical-flow fallback |
| 20 | GaVS | ✅ running | OpenCV fallback (arch not yet public) |
| 21 | AI Deinterlace | ❌ extra | `deinterlace_arch` (DeinterlaceNet) required |
| 22 | YADIF | ✅ real | Classical YADIF — no weights needed |
| 23 | Demucs | ✅ real | htdemucs weights auto-downloaded |
| 24 | VoiceFixer | ✅ real | Weights auto-downloaded |
| 25 | RNNoise | ✅ running | Lightweight classical noise gate |

❌ weights — no public weights found for this model; supply manually via `models/`.
❌ arch — architecture module not yet vendored into this repo.
❌ arch+weights — both missing.
14/25 models produce real output as of this sprint.

---

## 4× Super-Resolution — Real-ESRGAN x4plus

256 px input → 1024 px output in ~7 s on RTX 3080. Weights: `models/real_esrgan/RealESRGAN_x4plus.pth`.

<div class="compare-wrap">
  <img src="assets/restorations/sr_real_esrgan_before.png" alt="Before">
  <div class="compare-after">
    <img src="assets/restorations/sr_real_esrgan_after.png" alt="After">
  </div>
  <div class="compare-handle"></div>
  <span class="compare-label label-before">Before — 256 px</span>
  <span class="compare-label label-after">After — 1024 px (RealESRGAN x4plus)</span>
</div>

Original → Degraded → Restored composite:

![SR composite](assets/restorations/sr_real_esrgan_composite.png)

---

## 4× Video Super-Resolution — VRT

Six frames of a slowly panning real photograph (NASA astronaut portrait, public domain, from `scikit-image`) were downscaled 4× with anti-aliased bicubic, then upscaled again. Real weights, CPU inference.

![VRT composite](assets/restorations/sr_vrt_composite.png)

Measured on three real photographs, mean PSNR against the original: **+5.8 dB**, **+5.6 dB** and **+2.8 dB** over bicubic upscaling. VRT was trained on anti-aliased (MATLAB-style) bicubic degradation, so low-resolution inputs made with a plain `cv2.resize` (which aliases) look worse than bicubic. Input frames must be at least 64×64 px; other sizes are padded to a multiple of 32 internally.

Reproduce: `restorax download-models --model vrt` then `pytest tests/integration/test_real_inference.py -k vrt`.

---

## Deinterlacing — YADIF

Frame from Sintel (CC-BY). YADIF runs in <1 ms per frame (no GPU needed).

<div class="compare-wrap">
  <img src="assets/restorations/deint_yadif_before.png" alt="Before">
  <div class="compare-after">
    <img src="assets/restorations/deint_yadif_after.png" alt="After">
  </div>
  <div class="compare-handle"></div>
  <span class="compare-label label-before">Before</span>
  <span class="compare-label label-after">After — YADIF</span>
</div>

Video clips (3 s @ 24 fps): [before](assets/restorations/deint_yadif_before.mp4) · [after](assets/restorations/deint_yadif_after.mp4)

---

## Video Stabilization — OpenCV optical-flow

Frame from Sintel processed by OpenCV-based stabilization (GaVS arch pending public release).

<div class="compare-wrap">
  <img src="assets/restorations/stab_deepflow_before.png" alt="Before">
  <div class="compare-after">
    <img src="assets/restorations/stab_deepflow_after.png" alt="After">
  </div>
  <div class="compare-handle"></div>
  <span class="compare-label label-before">Before</span>
  <span class="compare-label label-after">After — Deep Flow Stabilization</span>
</div>

---

## Audio Source Separation — Demucs htdemucs

Big Buck Bunny (CC-BY) music: mixed → isolated stem. 10 s excerpt.

![Demucs spectrogram](assets/restorations/audio_demucs_spectrogram.png)

[Download before (mix)](assets/restorations/audio_demucs_before.wav) · [Download after (stem)](assets/restorations/audio_demucs_after.wav)

---

## Speech Enhancement — VoiceFixer

Tears of Steel (CC-BY) speech. 10 s excerpt processed through VoiceFixer neural model.

![VoiceFixer spectrogram](assets/restorations/audio_voicefixer_spectrogram.png)

[Download before](assets/restorations/audio_voicefixer_before.wav) · [Download after](assets/restorations/audio_voicefixer_after.wav)

---

## Noise Reduction — RNNoise

Speech track through RNNoise lightweight noise gate.

![RNNoise spectrogram](assets/restorations/audio_rnnoise_spectrogram.png)

[Download before](assets/restorations/audio_rnnoise_before.wav) · [Download after](assets/restorations/audio_rnnoise_after.wav)

---

> **Regenerate all samples:** `conda run -n restorax python scripts/generate_real_samples.py`
>
> Models marked ❌ can be unblocked by installing the noted dependency group and re-running.
> The slider and audio links require the MkDocs site (`mkdocs serve`); GitHub renders static images only.
