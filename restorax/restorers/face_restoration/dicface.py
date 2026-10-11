"""
DicFace — temporally-coherent video face restoration.

DicFace reformulates CodeFormer's discrete codebook lookup as Dirichlet-
distributed continuous variables and adds a temporal transformer over a
window of frames, giving smoother identity/detail consistency across a
video than single-frame blind face restorers (CodeFormer/GFPGAN).

Model source: https://github.com/fudan-generative-vision/DicFace
Paper: "DicFace: Dirichlet-Constrained Variational Codebook Learning for
        Temporally Coherent Video Face Restoration" (ICCV 2025)

Weight: downloaded from HuggingFace Hub (fudan-generative-ai/DicFace_model)
on first use — the "bfr_100k.pth" blind-face-restoration checkpoint.
"""

from __future__ import annotations

import logging
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

_HF_REPO = "fudan-generative-ai/DicFace_model"
_HF_WEIGHT_PATH = "CodeFormer/bfr_100k.pth"
_WEIGHT_FILE = "bfr_100k.pth"
_DEFAULT_FIDELITY = 0.5

# Checkpoint was trained with this fixed temporal window; the arch's
# position_emb_temporal parameter shape must match it exactly.
_FRAME_LENGTH = 5


class DicFaceRestorer(BaseRestorer):
    """
    Temporally-coherent blind face restoration (DicFace, ICCV 2025).

    Processes faces in non-overlapping windows of _FRAME_LENGTH frames so the
    codebook-lookup transformer has temporal context, reducing flicker versus
    CodeFormer/GFPGAN on video. Ideal for portrait videos and close-up
    interview footage.

    extra params:
      fidelity: float 0.0–1.0 (0=max enhancement, 1=max fidelity)
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
        return "dicface"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FACE_RESTORATION,
            input_color_space="bgr",
            output_color_space="bgr",
            requires_temporal=True,
            min_vram_gb=4.0,
            scale_factor=1,
            tags=["face_restoration", "dicface", "dictionary", "identity", "temporal", "iccv2025"],
        )

    def load(self, device: torch.device) -> None:
        try:
            from restorax.restorers.face_restoration.dicface_arch import (
                TemporalCodeFormerDirDistMultiScale,
            )
        except ImportError as exc:
            raise RestorerLoadError(
                "DicFace architecture module is required. "
                "Install with: pip install restorax[dicface]"
            ) from exc

        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        except ImportError as exc:
            raise RestorerLoadError(
                "facexlib is required for DicFace. Install with: pip install facexlib"
            ) from exc

        from restorax.config import settings

        weight_path = Path(settings.model_dir) / "dicface" / _WEIGHT_FILE
        if not weight_path.exists():
            try:
                from huggingface_hub import hf_hub_download

                weight_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=_HF_REPO, filename=_WEIGHT_FILE, local_dir=str(weight_path.parent)
                )
            except Exception as exc:
                raise RestorerLoadError(
                    f"Failed to download DicFace weights from {_HF_REPO}: {exc}"
                ) from exc

        try:
            net = TemporalCodeFormerDirDistMultiScale(
                dim_embed=512,
                n_head=8,
                n_layers=9,
                codebook_size=1024,
                connect_list=["32", "64", "128", "256"],
                frame_length=_FRAME_LENGTH,
            ).to(device)
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            net.load_state_dict(ckpt.get("params_ema", ckpt), strict=False)
            net.eval()
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load DicFace checkpoint: {exc}") from exc

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )

        self._net = net
        self._face_helper = face_helper
        self._device = device
        self._loaded = True
        logger.info("DicFace loaded on %s", device)

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
        helper = self._face_helper
        helper.clean_all()  # type: ignore[union-attr,attr-defined]
        helper.read_image(frame)  # type: ignore[union-attr,attr-defined]
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)  # type: ignore[union-attr,attr-defined]
        helper.align_warp_face()  # type: ignore[union-attr,attr-defined]
        if not helper.cropped_faces:  # type: ignore[union-attr,attr-defined]
            return frame
        restored_faces = []
        for face_bgr in helper.cropped_faces:  # type: ignore[union-attr,attr-defined]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            t = (
                torch.from_numpy(face_rgb)
                .float()
                .div(255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self._device)
            )
            with torch.inference_mode():
                out = self._net(t, w=fidelity)[0]  # type: ignore[operator]
            rgb = (
                out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).byte().cpu().numpy()
            )
            restored_faces.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        helper.add_restored_face(restored_faces)  # type: ignore[union-attr,attr-defined]
        helper.paste_faces_to_input_image()  # type: ignore[union-attr,attr-defined]
        return helper.output if helper.output is not None else frame  # type: ignore[union-attr,attr-defined]
