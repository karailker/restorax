"""Unit tests for HDRTVDMRestorer."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer

torch = pytest.importorskip("torch")


class TestHDRTVDMRestorerMeta:
    def test_name(self):
        assert HDRTVDMRestorer().name == "hdrtvdm"

    def test_capabilities_category(self):
        assert HDRTVDMRestorer().capabilities.category == RestorerCategory.HDR_CONVERSION

    def test_capabilities_scale_factor(self):
        assert HDRTVDMRestorer().capabilities.scale_factor == 1

    def test_capabilities_requires_temporal(self):
        assert HDRTVDMRestorer().capabilities.requires_temporal is False

    def test_capabilities_color_spaces(self):
        caps = HDRTVDMRestorer().capabilities
        assert caps.input_color_space == "rgb"
        assert caps.output_color_space == "rgb"

    def test_capabilities_tags_include_hdr(self):
        assert "hdr" in HDRTVDMRestorer().capabilities.tags


class TestHDRTVDMBuildModelRaisesWhenArchAbsent:
    def test_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "hdrtvdm_arch" in name:
                raise ImportError("No module named 'hdrtvdm_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="not vendored"):
                HDRTVDMRestorer._build_model(device)
