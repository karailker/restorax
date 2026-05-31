"""
CodeFormer face restorer.

Uses a VQ-codebook-based architecture for blind face restoration.
Achieves best balance of fidelity and perceptual quality among blind
face restoration methods (better than GFPGAN on degraded inputs).

Requires `codeformer-pytorch` PyPI package which bundles the model
architecture and facexlib for face detection/alignment.

Model source: https://github.com/sczhou/CodeFormer
Paper: "Towards Robust Blind Face Restoration with Codebook Lookup
        Transformer" (NeurIPS 2022)

Weight: downloaded from HuggingFace Hub on first use.
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
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
    ParamSpec,
)

logger = logging.getLogger(__name__)

_HF_REPO = "sczhou/CodeFormer"
_WEIGHT_FILE = "codeformer.pth"
_FACE_PARSE_MODEL = "parsing_parsenet.pth"

# fidelity weight: 0.0 = max enhancement, 1.0 = max fidelity to input
_DEFAULT_FIDELITY = 0.5


class CodeFormerRestorer(BaseRestorer):
    """
    Blind face restoration using CodeFormer.

    Detects faces in each frame, restores them individually, then blends
    the enhanced faces back into the full frame using face detection masks.

    The `fidelity` extra param (0.0–1.0) controls the trade-off:
      - 0.0: maximum enhancement (may hallucinate details on very degraded faces)
      - 1.0: maximum fidelity to input (preserves original structure)
      - 0.5: recommended default
    """

    PARAM_SCHEMA = [
        ParamSpec("fidelity", "float", _DEFAULT_FIDELITY, "Fidelity",
                  minimum=0.0, maximum=1.0, step=0.05,
                  help="0 = best quality, 1 = best identity preservation"),
    ]

    def __init__(self) -> None:
        self._net: torch.nn.Module | None = None
        self._face_helper: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "codeformer"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FACE_RESTORATION,
            input_color_space="bgr",   # facexlib expects BGR
            output_color_space="bgr",
            requires_temporal=False,
            min_vram_gb=4.0,
            scale_factor=1,
            tags=["face_restoration", "blind", "codeformer"],
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, device: torch.device) -> None:
        try:
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        except ImportError as exc:
            raise RestorerLoadError(
                "codeformer-pytorch and facexlib are required. "
                "Install with: pip install codeformer-pytorch facexlib"
            ) from exc

        weight_path = self._resolve_weight_path()
        logger.info("Loading CodeFormer from %s on %s", weight_path, device)

        net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RestorerLoadError(f"Failed to load CodeFormer checkpoint: {exc}") from exc

        net.load_state_dict(ckpt.get("params_ema", ckpt))
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

        self._net = net
        self._face_helper = face_helper
        self._device = device
        self._loaded = True
        logger.info("CodeFormer loaded successfully")

    def unload(self) -> None:
        del self._net
        self._net = None
        self._face_helper = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        """
        Detect and restore faces in a single BGR frame.

        Returns the frame with enhanced faces blended back in.
        If no faces are detected, returns the original frame unchanged.
        """
        assert self._net is not None and self._face_helper is not None
        assert self._device is not None

        fidelity = float(params.extra.get("fidelity", _DEFAULT_FIDELITY))

        helper = self._face_helper
        helper.clean_all()  # type: ignore[union-attr]
        helper.read_image(frame)  # type: ignore[union-attr]
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)  # type: ignore[union-attr]
        helper.align_warp_face()  # type: ignore[union-attr]

        # No faces detected — return unchanged
        if not helper.cropped_faces:  # type: ignore[union-attr]
            return frame

        restored_faces = []
        for cropped_face in helper.cropped_faces:  # type: ignore[union-attr]
            face_t = self._face_to_tensor(cropped_face)
            with torch.inference_mode():
                output = self._net(face_t, w=fidelity, adain=True)[0]
            restored = self._tensor_to_face(output)
            restored_faces.append(restored)

        helper.add_restored_face(restored_faces)  # type: ignore[union-attr]
        helper.paste_faces_to_input_image()  # type: ignore[union-attr]
        result = helper.output  # type: ignore[union-attr]

        return result if result is not None else frame

    # ── Internal ──────────────────────────────────────────────────────────────

    def _face_to_tensor(self, face_bgr: np.ndarray) -> torch.Tensor:
        assert self._device is not None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(face_rgb).float().div(255.0).permute(2, 0, 1)
        return t.unsqueeze(0).to(self._device)

    @staticmethod
    def _tensor_to_face(tensor: torch.Tensor) -> np.ndarray:
        face = tensor.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).byte()
        face_bgr = cv2.cvtColor(face.cpu().numpy(), cv2.COLOR_RGB2BGR)
        return face_bgr

    def _resolve_weight_path(self) -> Path:
        from restorax.config import settings

        model_dir = Path(settings.model_dir) / "codeformer"
        weight_path = model_dir / _WEIGHT_FILE
        if not weight_path.exists():
            weight_path = self._download_weights(model_dir)
        return weight_path

    @staticmethod
    def _download_weights(model_dir: Path) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RestorerLoadError("huggingface_hub required.") from exc

        logger.info("Downloading CodeFormer weights…")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"CodeFormer/{_WEIGHT_FILE}",
            local_dir=str(model_dir),
        )
        return Path(path)
