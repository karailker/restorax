"""Unit tests for AIDeinterlaceRestorer."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer

torch = pytest.importorskip("torch")


class TestAIDeinterlaceRestorerMeta:
    def test_name(self):
        assert AIDeinterlaceRestorer().name == "ai_deinterlace"

    def test_capabilities_category(self):
        assert AIDeinterlaceRestorer().capabilities.category == RestorerCategory.DEINTERLACING

    def test_capabilities_scale_factor(self):
        assert AIDeinterlaceRestorer().capabilities.scale_factor == 1

    def test_capabilities_requires_temporal(self):
        assert AIDeinterlaceRestorer().capabilities.requires_temporal is False

    def test_capabilities_color_spaces(self):
        caps = AIDeinterlaceRestorer().capabilities
        assert caps.input_color_space == "rgb"
        assert caps.output_color_space == "rgb"


class TestAIDeinterlaceLoadArchRaisesWhenArchAbsent:
    def test_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "deinterlace_arch" in name:
                raise ImportError("No module named 'deinterlace_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="deinterlace_arch"):
                AIDeinterlaceRestorer._load_arch(device)

    def test_load_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "deinterlace_arch" in name:
                raise ImportError("No module named 'deinterlace_arch'")
            return real_import(name, *args, **kwargs)

        restorer = AIDeinterlaceRestorer()
        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError):
                restorer.load(device)

    def test_not_loaded_after_failed_load(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "deinterlace_arch" in name:
                raise ImportError("No module named 'deinterlace_arch'")
            return real_import(name, *args, **kwargs)

        restorer = AIDeinterlaceRestorer()
        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError):
                restorer.load(device)
        assert not restorer.is_loaded
