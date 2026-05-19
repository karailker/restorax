"""Unit tests for CodeFormerPlusPlusRestorer — no GPU, no real weights."""
from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer

torch = pytest.importorskip("torch")


class TestCodeFormerPlusPlusMeta:
    def test_name(self):
        assert CodeFormerPlusPlusRestorer().name == "codeformer_pp"

    def test_capabilities_category(self):
        assert CodeFormerPlusPlusRestorer().capabilities.category == RestorerCategory.FACE_RESTORATION

    def test_capabilities_scale_factor(self):
        assert CodeFormerPlusPlusRestorer().capabilities.scale_factor == 1

    def test_capabilities_requires_temporal(self):
        assert CodeFormerPlusPlusRestorer().capabilities.requires_temporal is False

    def test_capabilities_color_spaces(self):
        caps = CodeFormerPlusPlusRestorer().capabilities
        assert caps.input_color_space == "bgr"
        assert caps.output_color_space == "bgr"


class TestCodeFormerPlusPlusBuildModelRaisesWhenArchAbsent:
    def test_raises_restorer_load_error_on_missing_arch(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "codeformer_pp_arch" in name:
                raise ImportError("No module named 'codeformer_pp_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="codeformer_pp_arch"):
                CodeFormerPlusPlusRestorer._build_model(device)

    def test_raises_restorer_load_error_not_returns_none(self):
        """Ensure _build_model never silently returns None - it must raise."""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "codeformer_pp_arch" in name:
                raise ImportError("No module named 'codeformer_pp_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError):
                result = CodeFormerPlusPlusRestorer._build_model(device)
                # Should never reach here — _build_model must not return None
                assert result != (None, None)
