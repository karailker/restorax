"""Unit tests for VRTRestorer."""

from __future__ import annotations

import pytest

from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.vrt import VRTRestorer

torch = pytest.importorskip("torch")


class TestVRTRestorerMeta:
    def test_name(self):
        r = VRTRestorer()
        assert r.name == "vrt_x4"

    def test_capabilities_category(self):
        caps = VRTRestorer().capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION

    def test_capabilities_scale_factor(self):
        caps = VRTRestorer().capabilities
        assert caps.scale_factor == 4

    def test_capabilities_requires_temporal(self):
        caps = VRTRestorer().capabilities
        assert caps.requires_temporal is True


# ponytail: VRT arch is vendored (vrt_arch.py, no basicsr dep), so "basicsr
# absent" can no longer trigger an arch-load failure — that test class was
# deleted. The real guard now is the no-weights RestorerLoadError in _build_model.


class TestVRTSequencePadding:
    """The model sees a multiple of the 6-frame window; padding is dropped again."""

    @staticmethod
    def _restorer_with_fake_model(seen: list[int], sizes: list[tuple[int, int]] | None = None):
        import numpy as np

        r = VRTRestorer()

        def fake_model(video):  # 1 T C H W -> 1 T C 4H 4W
            seen.append(video.shape[1])
            if sizes is not None:
                sizes.append((video.shape[-2], video.shape[-1]))
            b, t, c, h, w = video.shape
            return torch.nn.functional.interpolate(
                video.reshape(b * t, c, h, w), scale_factor=4, mode="nearest"
            ).reshape(b, t, c, h * 4, w * 4)

        r._model = fake_model  # type: ignore[assignment]
        r._device = torch.device("cpu")
        r._loaded = True
        assert np  # keep numpy import local to the helper
        return r

    @pytest.mark.parametrize("n_frames,model_frames", [(1, 6), (5, 6), (6, 6), (7, 12), (12, 12)])
    def test_pads_to_window_multiple(self, n_frames, model_frames):
        import numpy as np

        from restorax.core.restorer import RestorerParams

        seen: list[int] = []
        r = self._restorer_with_fake_model(seen)
        frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n_frames)]
        out = r.process_sequence(frames, RestorerParams(half_precision=False))
        assert seen == [model_frames]
        assert len(out) == n_frames
        assert out[0].shape == (32, 32, 3)
        # output frame i comes from input frame i (padding is dropped, not shuffled)
        assert [int(o[0, 0, 0]) for o in out] == list(range(n_frames))


def test_weight_url_is_a_github_release_asset():
    from restorax.restorers.super_resolution import vrt

    assert vrt._WEIGHT_URL.startswith("https://github.com/JingyunLiang/VRT/releases/download/")
    assert vrt._WEIGHT_URL.endswith(vrt._WEIGHT_FILE)


def test_catalog_entry_matches_restorer():
    from restorax.models_catalog import CATALOG_BY_NAME
    from restorax.restorers.super_resolution import vrt

    entry = CATALOG_BY_NAME["vrt"]
    assert entry.weight_files == [vrt._WEIGHT_FILE]
    assert entry.urls == {vrt._WEIGHT_FILE: vrt._WEIGHT_URL}


@pytest.mark.parametrize(
    "h,w,padded",
    [
        (32, 48, (64, 64)),
        (63, 64, (64, 64)),
        (64, 64, (64, 64)),
        (70, 90, (96, 96)),
        (96, 96, (96, 96)),
    ],
)
def test_spatial_padding_is_applied_and_cropped(h, w, padded):
    import numpy as np

    from restorax.core.restorer import RestorerParams

    sizes: list[tuple[int, int]] = []
    r = TestVRTSequencePadding._restorer_with_fake_model([], sizes)
    frames = [np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)] * 6
    out = r.process_sequence(frames, RestorerParams(half_precision=False))
    assert sizes == [padded]  # multiples of 32, at least 64
    assert out[0].shape == (h * 4, w * 4, 3)  # cropped back to 4x the original
