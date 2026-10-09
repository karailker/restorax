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
@pytest.mark.requires_assets
def test_vrt_upscales_sequence(test_assets):
    """VRT produces 4× output for a 7-frame temporal window."""
    import cv2

    from restorax.core.restorer import RestorerParams
    from restorax.restorers.super_resolution.vrt import VRTRestorer

    img_path = test_assets / "set5" / "butterfly.png"
    raw = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    # Crop to 64×64 to keep inference fast on CPU
    frame = raw[:64, :64]
    frames = [frame] * 7

    restorer = VRTRestorer()
    restorer.load(torch.device("cpu"))
    outs = restorer.process_sequence(frames, RestorerParams(scale=4, half_precision=False))
    restorer.unload()

    assert len(outs) == 7
    assert outs[0].shape == (256, 256, 3)


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
@pytest.mark.requires_assets
def test_ddcolor_colorizes_grayscale(test_assets):
    """DDColor produces an RGB colorized output from a grayscale input."""
    import cv2

    from restorax.core.restorer import RestorerParams
    from restorax.restorers.colorization.ddcolor import DDColorRestorer

    img_path = test_assets / "set5" / "bird.png"
    frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)

    restorer = DDColorRestorer()
    restorer.load(torch.device("cpu"))
    out = restorer.process_frame(gray, RestorerParams(scale=1, half_precision=False))
    restorer.unload()

    assert out.shape == gray.shape
    assert out.dtype == np.uint8
