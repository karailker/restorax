"""Unit tests for DDColorRestorer."""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.colorization.ddcolor import DDColorRestorer

torch = pytest.importorskip("torch")


class TestDDColorRestorerMeta:
    def test_name(self):
        assert DDColorRestorer().name == "ddcolor"

    def test_capabilities_category(self):
        assert DDColorRestorer().capabilities.category == RestorerCategory.COLORIZATION

    def test_capabilities_scale_factor(self):
        assert DDColorRestorer().capabilities.scale_factor == 1

    def test_capabilities_requires_temporal(self):
        assert DDColorRestorer().capabilities.requires_temporal is False

    def test_capabilities_color_spaces(self):
        caps = DDColorRestorer().capabilities
        assert caps.input_color_space == "rgb"
        assert caps.output_color_space == "rgb"


class TestDDColorBuildModelRaisesWhenArchAbsent:
    def test_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "ddcolor_arch" in name:
                raise ImportError("No module named 'ddcolor_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="DDColor unavailable"):
                DDColorRestorer._build_model(Path("/fake/weights.pth"), device)
