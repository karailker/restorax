"""
Tier 3: Real inference tests.

All tests are skipped automatically unless:
  - @pytest.mark.requires_weights("model"): weights dir exists
  - @pytest.mark.requires_assets: tests/assets/ has been populated

Run manually after downloading weights:
  restorax download-models --model real_esrgan
  python -m pytest tests/integration/test_real_inference.py -m requires_weights -v
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.requires_weights("real_esrgan")
@pytest.mark.requires_assets
def test_real_esrgan_upscales_set5_butterfly(test_assets):
    """RealESRGAN produces 4× output on a real image."""
    import cv2

    from restorax.core.restorer import RestorerParams
    from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer

    img_path = test_assets / "set5" / "butterfly.png"
    frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    restorer = RealESRGANx4Restorer()
    restorer.load(torch.device("cpu"))
    out = restorer.process_frame(frame, RestorerParams(scale=4, half_precision=False))
    restorer.unload()

    assert out.shape == (h * 4, w * 4, 3)
    assert out.dtype == np.uint8


@pytest.mark.requires_weights("vrt")
def test_vrt_beats_bicubic_on_real_photos():
    """VRT with the official REDS weights must clearly beat bicubic upscaling.

    Frames are degraded with anti-aliased bicubic downscaling (the protocol VRT
    was trained for; naive cv2 downscaling aliases and makes any SR look bad).
    """
    cv2 = pytest.importorskip("cv2")
    skimage_data = pytest.importorskip("skimage.data")
    pil = pytest.importorskip("PIL.Image")

    from restorax.core.restorer import RestorerParams
    from restorax.restorers.super_resolution.vrt import VRTRestorer

    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return float(10 * np.log10(255**2 / mse))

    height, width = 256, 256
    image = np.ascontiguousarray(skimage_data.astronaut())
    gt = [image[100 + i : 100 + i + height, 120 + 2 * i : 120 + 2 * i + width] for i in range(6)]
    lr = [np.asarray(pil.fromarray(f).resize((width // 4, height // 4), pil.BICUBIC)) for f in gt]

    restorer = VRTRestorer()
    restorer.load(torch.device("cpu"))
    out = restorer.process_sequence(lr, RestorerParams(half_precision=False))
    restorer.unload()

    bicubic = [cv2.resize(f, (width, height), interpolation=cv2.INTER_CUBIC) for f in lr]
    gain = np.mean([psnr(o, g) - psnr(b, g) for o, b, g in zip(out, bicubic, gt, strict=True)])
    assert out[0].shape == (height, width, 3) and out[0].dtype == np.uint8
    assert gain > 1.5, f"VRT only gained {gain:.2f} dB over bicubic"


@pytest.mark.requires_weights("waifu2x")
@pytest.mark.requires_assets
def test_waifu2x_upscales_set5(test_assets):
    """Waifu2x produces 2× output on a real image."""
    import cv2

    from restorax.core.restorer import RestorerParams
    from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer

    img_path = test_assets / "set5" / "baby.png"
    frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    restorer = Waifu2xRestorer()
    restorer.load(torch.device("cpu"))
    out = restorer.process_frame(frame, RestorerParams(scale=2, half_precision=False))
    restorer.unload()

    assert out.shape[0] >= h and out.shape[1] >= w


@pytest.mark.requires_weights("ddcolor")
def test_ddcolor_restores_plausible_color():
    """DDColor must add real chroma and move a gray photo closer to the color original.

    Uses the bundled scikit-image photographs (no downloads), converts them to gray and
    checks that the colorized result is more saturated than gray and has a lower error
    against the original than the gray input does.
    """
    cv2 = pytest.importorskip("cv2")
    skimage_data = pytest.importorskip("skimage.data")

    from restorax.core.restorer import RestorerParams
    from restorax.restorers.colorization.ddcolor import DDColorRestorer

    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return float(10 * np.log10(255**2 / mse))

    restorer = DDColorRestorer()
    restorer.load(torch.device("cpu"))
    try:
        gains = []
        for image in (skimage_data.astronaut(), skimage_data.coffee(), skimage_data.chelsea()):
            color = np.ascontiguousarray(image[:256, :256])
            gray = np.repeat(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1)
            out = restorer.process_frame(gray, RestorerParams(half_precision=False))
            assert out.shape == gray.shape and out.dtype == np.uint8
            chroma = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[:, :, 1:].astype(float) - 128
            assert np.abs(chroma).mean() > 3, "output is still (almost) gray"
            gains.append(psnr(out, color) - psnr(gray, color))
        assert np.mean(gains) > 0.5, f"colorization did not approach the original: {gains}"
    finally:
        restorer.unload()
