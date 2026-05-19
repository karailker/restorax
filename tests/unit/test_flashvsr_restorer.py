"""Unit tests for FlashVSRRestorer."""
from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer

torch = pytest.importorskip("torch")


class TestFlashVSRRestorerMeta:
    def test_name(self):
        assert FlashVSRRestorer().name == "flashvsr_x4"

    def test_capabilities_scale_factor(self):
        assert FlashVSRRestorer().capabilities.scale_factor == 4

    def test_capabilities_requires_temporal(self):
        assert FlashVSRRestorer().capabilities.requires_temporal is True

    def test_capabilities_category(self):
        assert FlashVSRRestorer().capabilities.category == RestorerCategory.SUPER_RESOLUTION


class TestFlashVSRBuildModelRaisesWhenArchAbsent:
    def test_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "flashvsr_arch" in name:
                raise ImportError("No module named 'flashvsr_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="FlashVSR arch module is not available"):
                FlashVSRRestorer._build_model(device)
