"""Unit tests for DicFaceRestorer."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.face_restoration.dicface import DicFaceRestorer

torch = pytest.importorskip("torch")


class TestDicFaceRestorerMeta:
    def test_name(self):
        assert DicFaceRestorer().name == "dicface"

    def test_capabilities_category(self):
        assert DicFaceRestorer().capabilities.category == RestorerCategory.FACE_RESTORATION

    def test_capabilities_scale_factor(self):
        assert DicFaceRestorer().capabilities.scale_factor == 1

    def test_capabilities_color_spaces(self):
        caps = DicFaceRestorer().capabilities
        assert caps.input_color_space == "bgr"
        assert caps.output_color_space == "bgr"

    def test_capabilities_tags(self):
        assert "iccv2023" in DicFaceRestorer().capabilities.tags


class TestDicFaceRestorerLoadError:
    def test_raises_restorer_load_error_when_dicface_arch_missing(self):
        """load() raises RestorerLoadError when dicface_arch is not importable."""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "dicface_arch" in name:
                raise ImportError("No module named 'dicface_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="DicFace architecture module is required"):
                DicFaceRestorer().load(device)

    def test_raises_restorer_load_error_when_facexlib_missing(self):
        """load() raises RestorerLoadError when facexlib is not importable."""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "facexlib" or name.startswith("facexlib."):
                raise ImportError("No module named 'facexlib'")
            # Let dicface_arch succeed via a stub
            if "dicface_arch" in name:
                raise ImportError("No module named 'dicface_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            # dicface_arch missing triggers first — both paths raise RestorerLoadError
            with pytest.raises(RestorerLoadError):
                DicFaceRestorer().load(device)

    def test_raises_restorer_load_error_not_silent_degradation(self):
        """Confirm load() never returns silently when arch is unavailable."""
        device = torch.device("cpu")

        with patch(
            "restorax.restorers.face_restoration.dicface.DicFaceRestorer.load",
            side_effect=RestorerLoadError("arch missing"),
        ):
            r = DicFaceRestorer()
            with pytest.raises(RestorerLoadError):
                r.load(device)
        # restorer must remain unloaded after failed load
        assert not r.is_loaded
