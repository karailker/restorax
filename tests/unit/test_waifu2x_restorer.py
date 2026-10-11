"""Unit tests for Waifu2xRestorer."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer

torch = pytest.importorskip("torch")


class TestWaifu2xRestorerMeta:
    def test_name(self):
        assert Waifu2xRestorer().name == "waifu2x_x2"

    def test_capabilities_category(self):
        assert Waifu2xRestorer().capabilities.category == RestorerCategory.SUPER_RESOLUTION

    def test_capabilities_scale_factor(self):
        assert Waifu2xRestorer().capabilities.scale_factor == 2


class TestWaifu2xBuildModelRaisesOnMissingArch:
    def test_raises_restorer_load_error_on_arch_import_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "waifu2x_arch" in name:
                raise ImportError("No module named 'waifu2x_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="waifu2x_arch module not found"):
                Waifu2xRestorer._build_model(device)


class TestWaifu2xBuildModelRaisesOnMissingWeights:
    def test_raises_restorer_load_error_when_py7zr_unavailable(self, tmp_path):
        """When weights are absent and py7zr is missing, RestorerLoadError explains why."""

        class _FakeUpConvNet(torch.nn.Module):
            def __init__(self, scale: int) -> None:
                super().__init__()

        fake_arch = MagicMock()
        fake_arch.UpConvNet = _FakeUpConvNet

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "waifu2x_arch" in name:
                return fake_arch
            if name == "py7zr":
                raise ImportError("No module named 'py7zr'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        # Point model_dir at tmp_path so weight_path will not exist
        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("restorax.config.settings.model_dir", str(tmp_path)),
        ):
            with pytest.raises(RestorerLoadError, match="py7zr is required"):
                Waifu2xRestorer._build_model(device)


class TestWaifu2xArch:
    def test_upconvnet_is_instantiable_module_with_parameters(self):
        from restorax.restorers.super_resolution.waifu2x_arch import UpConvNet

        model = UpConvNet(scale=2)
        assert isinstance(model, torch.nn.Module)
        assert len(list(model.parameters())) > 0
