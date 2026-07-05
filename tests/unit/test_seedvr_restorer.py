from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.seedvr import SeedVRRestorer


class TestSeedVRMeta:
    def test_name(self):
        assert SeedVRRestorer().name == "seedvr"

    def test_category(self):
        assert SeedVRRestorer().capabilities.category == RestorerCategory.SUPER_RESOLUTION

    def test_requires_temporal(self):
        assert SeedVRRestorer().capabilities.requires_temporal is True

    def test_scale_factor(self):
        assert SeedVRRestorer().capabilities.scale_factor == 4


class TestSeedVRLoadRaisesWhenDiffusersAbsent:
    def test_raises_restorer_load_error_on_missing_diffusers(self, tmp_path):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "diffusers" or name.startswith("diffusers."):
                raise ImportError("No module named 'diffusers'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("restorax.config.settings.model_dir", str(tmp_path)),
        ):
            with pytest.raises(RestorerLoadError):
                SeedVRRestorer().load(torch.device("cpu"))


class TestSeedVRArchShim:
    """SeedVR's real weights (causal VAE + Swin-MMDiT) aren't vendored — only
    an adapter shim is. from_pretrained must fail loudly, not fake a pipeline."""

    def test_from_pretrained_raises_import_error(self):
        pytest.importorskip("diffusers")
        from restorax.restorers.super_resolution.seedvr_arch import SeedVRPipeline

        with pytest.raises(ImportError):
            SeedVRPipeline.from_pretrained("unused")
