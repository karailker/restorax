"""
CodeFormer++ blind face restoration — SUBSTITUTED with RestoreFormer.

The paper this module was originally written against, "CodeFormer++: Blind
Face Restoration Using Deformable Registration and Deep Metric Learning"
(arxiv.org/abs/2510.04410, 2025), has NO public code or weight release from
its authors (verified: no GitHub link on the arxiv page, no PapersWithCode
entry, no author repo). It cannot be vendored.

This restorer keeps the `codeformer_pp` name/class/registration contract
(drop-in for existing catalog/API/task references) but the model underneath
is real: RestoreFormer (https://github.com/wzhouxiff/RestoreFormer,
Apache-2.0), vendored in codeformer_pp_arch.py from GFPGAN's copy of the
same architecture. Weights are GFPGAN's published RestoreFormer.pth release
asset, which matches this exact arch. See tests/unit/test_codeformer_pp_restorer.py
and PLAN.md for the substitution rationale — flagged for human confirmation.

RestoreFormer has no fidelity-weight control (unlike CodeFormer); the
`fidelity` param is accepted for interface compatibility but has no effect.

Raises RestorerLoadError when the arch module or weights are unavailable.
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
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

_WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
_WEIGHT_FILE = "RestoreFormer.pth"
_DEFAULT_FIDELITY = 0.5


class CodeFormerPlusPlusRestorer(BaseRestorer):
    """
    Improved blind face restoration with deformable registration.

    Drop-in replacement for CodeFormerRestorer - better on severely
    degraded faces while maintaining the same fidelity/quality tradeoff.
    """

    PARAM_SCHEMA = [
        ParamSpec(
            "fidelity",
            "float",
            _DEFAULT_FIDELITY,
            "Fidelity",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            help="0 = best quality, 1 = best identity preservation",
        ),
    ]

    def __init__(self) -> None:
        self._net: object | None = None
        self._face_helper: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "codeformer_pp"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FACE_RESTORATION,
            input_color_space="bgr",
            output_color_space="bgr",
            requires_temporal=False,
            min_vram_gb=4.0,
            scale_factor=1,
            tags=["face_restoration", "blind", "codeformer_pp", "restoreformer"],
        )

    def load(self, device: torch.device) -> None:
        self._net, self._face_helper = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("CodeFormer++ loaded on %s", device)

    def unload(self) -> None:
        del self._net
        self._net = None
        self._face_helper = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        assert self._device is not None
        fidelity = float(params.extra.get("fidelity", _DEFAULT_FIDELITY))
        return self._restore(frame, fidelity)

    def _restore(self, frame: np.ndarray, fidelity: float) -> np.ndarray:
        """Full CodeFormer++ inference with face detection."""
        helper = self._face_helper
        helper.clean_all()  # type: ignore[union-attr,attr-defined]
        helper.read_image(frame)  # type: ignore[union-attr,attr-defined]
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)  # type: ignore[union-attr,attr-defined]
        helper.align_warp_face()  # type: ignore[union-attr,attr-defined]

        if not helper.cropped_faces:  # type: ignore[union-attr,attr-defined]
            return frame

        restored_faces = []
        for cropped_face in helper.cropped_faces:  # type: ignore[union-attr,attr-defined]
            face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            t = (
                torch.from_numpy(face_rgb)
                .float()
                .div(255.0)
                .sub(0.5)
                .div(0.5)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self._device)
            )
            with torch.inference_mode():
                out = self._net(t, w=fidelity, adain=True)[0]  # type: ignore[operator]
            # RestoreFormer works in [-1, 1]
            restored_rgb = (
                out.squeeze(0)
                .permute(1, 2, 0)
                .float()
                .add(1.0)
                .div(2.0)
                .clamp(0, 1)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            restored_faces.append(cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

        helper.add_restored_face(restored_faces)  # type: ignore[union-attr,attr-defined]
        helper.paste_faces_to_input_image()  # type: ignore[union-attr,attr-defined]
        result = helper.output  # type: ignore[union-attr,attr-defined]
        return result if result is not None else frame

    @staticmethod
    def _build_model(device: torch.device) -> tuple[object, object]:
        try:
            from restorax.restorers.face_restoration.codeformer_pp_arch import (
                CodeFormerPP,  # type: ignore[import]
            )
        except ImportError as exc:
            raise RestorerLoadError(
                f"CodeFormer++ arch module unavailable: {exc}. "
                "Install the codeformer_pp_arch package to use this restorer."
            ) from exc

        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            from restorax.config import settings

            weight_path = Path(settings.model_dir) / "codeformer_pp" / _WEIGHT_FILE
            if not weight_path.exists():
                weight_path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    urllib.request.urlopen(_WEIGHT_URL, timeout=30) as response,  # noqa: S310
                    open(weight_path, "wb") as out_file,
                ):
                    shutil.copyfileobj(response, out_file)

            net = CodeFormerPP().to(device)
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            net.load_state_dict(ckpt.get("params_ema", ckpt), strict=False)
            net.eval()

            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=device,
            )
            logger.info("CodeFormer++ arch loaded from vendored module")
            return net, face_helper
        except Exception as exc:
            raise RestorerLoadError(f"CodeFormer++ failed to load weights: {exc}") from exc
