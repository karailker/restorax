"""Unit tests for Waifu2xRestorer."""
from __future__ import annotations

import builtins
from pathlib import Path
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
    def test_raises_restorer_load_error_on_import_error(self):
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
    def test_raises_restorer_load_error_when_weights_missing_and_download_fails(self, tmp_path):
        """When weights file is absent and download raises, RestorerLoadError is raised."""

        # Provide a minimal UpConvNet stand-in so the arch import succeeds
        class _FakeUpConvNet(torch.nn.Module):
            def __init__(self, scale: int) -> None:
                super().__init__()

        fake_arch = MagicMock()
        fake_arch.UpConvNet = _FakeUpConvNet

        # Weight path points to a file that doesn't exist
        weight_path = tmp_path / "waifu2x" / "waifu2x_x2.pth"

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "waifu2x_arch" in name:
                return fake_arch
            if name == "huggingface_hub" or name.startswith("huggingface_hub"):
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch(
                "restorax.restorers.super_resolution.waifu2x.Path",
                return_value=weight_path,
            ),
        ):
            with pytest.raises(RestorerLoadError):
                Waifu2xRestorer._build_model(device)
