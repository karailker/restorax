"""
DLSS 5 Neural Rendering enhancement via an external Visual Enhancer install.

NVIDIA DLSS 5 ("3D-guided neural rendering") is a proprietary, RTX-only
technology. RestoraX does **not** bundle, copy or re-implement it. This
restorer is a thin adapter that drives the command-line interface of
`Visual Enhancer <https://github.com/Merserk/dlss5-visual-enhancer>`_
(``VE_CLI.exe``), which the user must download and install themselves from its
official releases and which is covered by its own licence and NVIDIA's SDK
licence.

Requirements (all on the user's side):
  * A compatible NVIDIA RTX GPU and current driver
  * Visual Enhancer's portable release extracted somewhere
  * ``RESTORAX_VE_CLI`` env var (or ``extra["ve_cli"]``) pointing at ``VE_CLI.exe``,
    or ``VE_CLI.exe`` on ``PATH``
  * Windows 11: nothing else. Linux (EXPERIMENTAL): Visual Enhancer is a Windows
    application (Direct3D 12 + NVIDIA's Windows NGX DLLs) and cannot run natively,
    so it is launched through Wine/Proton. Set ``RESTORAX_VE_LAUNCHER`` (or
    ``extra["launcher"]``) to the launcher command, e.g. ``wine`` or
    ``env WINEPREFIX=/path/to/prefix wine``; ``wine`` on ``PATH`` is used by default.
    Frame paths are converted with ``winepath -w``. This path has not been verified
    against real hardware; success depends on your Wine/Proton + NVIDIA driver setup.

Each frame is round-tripped through a temporary PNG, so this restorer is meant
for stills, keyframes and short clips. For long videos, run ``VE_CLI.exe``
directly on the file. The output keeps the input resolution; use a separate
RestoraX super-resolution stage for scaling.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import (
    BaseRestorer,
    ParamSpec,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)

_ENV_VAR = "RESTORAX_VE_CLI"
_LAUNCHER_ENV_VAR = "RESTORAX_VE_LAUNCHER"
_DEFAULT_TIMEOUT_S = 300.0
_STYLES = ("0", "1", "2")


class DLSS5VisualEnhancerRestorer(BaseRestorer):
    """Neural-rendering enhancement using DLSS 5 through an external Visual Enhancer CLI."""

    PARAM_SCHEMA = [
        ParamSpec(
            "style",
            "enum",
            "0",
            "NR style",
            choices=_STYLES,
            help="DLSS 5 Neural Rendering style (Visual Enhancer Style 0/1/2)",
        ),
        ParamSpec(
            "strength",
            "float",
            1.0,
            "NR intensity",
            minimum=0.0,
            maximum=2.0,
            step=0.05,
            help="Neural Rendering intensity; 1.0 is the Visual Enhancer default",
        ),
    ]

    def __init__(self) -> None:
        self._cli: Path | None = None
        self._launcher: list[str] = []
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "dlss5_visual_enhancer"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.ENHANCEMENT,
            input_color_space="rgb",
            output_color_space="rgb",
            requires_temporal=False,
            min_vram_gb=8.0,
            scale_factor=1,
            tags=[
                "enhancement",
                "dlss5",
                "neural_rendering",
                "nvidia",
                "rtx",
                "external",
                "windows",
            ],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        self._launcher = self._resolve_launcher()
        self._cli = self._locate_cli()
        self._device = device
        self._loaded = True
        logger.info("DLSS 5 Visual Enhancer CLI: %s", self._cli)

    def unload(self) -> None:
        self._loaded = False
        self._cli = None
        self._launcher = []
        self._device = None

    @staticmethod
    def _resolve_launcher(override: str | None = None) -> list[str]:
        """Command prefix used to start VE_CLI.exe: empty on Windows, Wine/Proton elsewhere."""
        if sys.platform == "win32":
            return []
        configured = override or os.environ.get(_LAUNCHER_ENV_VAR)
        if configured:
            return shlex.split(configured)
        if shutil.which("wine"):
            return ["wine"]
        raise RestorerLoadError(
            "Visual Enhancer is a Windows application. On Linux, install Wine/Proton and "
            f"set {_LAUNCHER_ENV_VAR} (e.g. 'wine') to run it (experimental)."
        )

    def _native_path(self, path: Path) -> str:
        """Path as seen by the launched process (winepath conversion under Wine)."""
        if not self._launcher:
            return str(path)
        winepath = shutil.which("winepath")
        if winepath is None:
            return str(path)
        try:
            out = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [winepath, "-w", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"dlss5_visual_enhancer: winepath failed for {path}: {exc}") from exc
        return out or str(path)

    @staticmethod
    def _locate_cli(override: str | None = None) -> Path:
        candidate = override or os.environ.get(_ENV_VAR) or shutil.which("VE_CLI.exe")
        if not candidate:
            raise RestorerLoadError(
                f"Visual Enhancer CLI not found. Install it from "
                f"https://github.com/Merserk/dlss5-visual-enhancer/releases and set "
                f"{_ENV_VAR} to the path of VE_CLI.exe."
            )
        path = Path(candidate).expanduser()
        if not path.is_file():
            raise RestorerLoadError(f"Visual Enhancer CLI not found at: {path}")
        return path

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        if not self._loaded or self._cli is None:
            raise RuntimeError("dlss5_visual_enhancer: load() must be called first")
        extra = params.extra
        style = str(extra.get("style", "0"))
        if style not in _STYLES:
            raise ValueError(f"style must be one of {_STYLES}, got {style!r}")
        strength = float(extra.get("strength", 1.0))
        if not 0.0 <= strength <= 2.0:
            raise ValueError(f"strength must be within 0..2, got {strength}")

        with tempfile.TemporaryDirectory(prefix="restorax_ve_") as tmp:
            src = Path(tmp) / "in.png"
            dst = Path(tmp) / "out.png"
            if not cv2.imwrite(str(src), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("dlss5_visual_enhancer: failed to write temporary input frame")
            cmd = [
                *self._launcher,
                str(self._cli),
                "render",
                self._native_path(src),
                "-o",
                self._native_path(dst),
                "--style",
                style,
                "--strength",
                str(strength),
                "--format",
                "png",
                "--quiet",
                "--json",
            ]
            if extra.get("gpu") is not None:
                cmd += ["--gpu", str(extra["gpu"])]
            if extra.get("preset"):
                cmd += ["--preset", self._native_path(Path(str(extra["preset"])))]
            self._run(cmd, float(extra.get("timeout", _DEFAULT_TIMEOUT_S)))

            out = cv2.imread(str(dst), cv2.IMREAD_COLOR)
            if out is None:
                raise RuntimeError(
                    "dlss5_visual_enhancer: Visual Enhancer produced no output image"
                )
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if out.shape != frame.shape:
            raise RuntimeError(
                f"dlss5_visual_enhancer changed frame size {frame.shape[:2]} -> {out.shape[:2]}; "
                "disable scaling in the Visual Enhancer preset (scale_factor is 1)"
            )
        return out

    @staticmethod
    def _run(cmd: list[str], timeout: float) -> None:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"dlss5_visual_enhancer timed out after {timeout:.0f}s") from exc
        if proc.returncode == 0:
            return
        detail = (proc.stderr or proc.stdout or "").strip()
        with contextlib.suppress(ValueError, AttributeError):
            detail = json.loads(proc.stdout).get("error", detail)
        raise RuntimeError(f"Visual Enhancer exited with status {proc.returncode}: {detail[:500]}")
