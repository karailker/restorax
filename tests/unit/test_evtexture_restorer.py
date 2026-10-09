"""Unit tests for EvTextureRestorer."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.evtexture import EvTextureRestorer

torch = pytest.importorskip("torch")


class TestEvTextureRestorerMeta:
    def test_name(self):
        assert EvTextureRestorer().name == "evtexture_x4"

    def test_capabilities_scale_factor(self):
        assert EvTextureRestorer().capabilities.scale_factor == 4

    def test_capabilities_requires_temporal(self):
        assert EvTextureRestorer().capabilities.requires_temporal is True

    def test_capabilities_category(self):
        assert EvTextureRestorer().capabilities.category == RestorerCategory.SUPER_RESOLUTION


class TestEvTextureBuildModelRaisesWhenArchAbsent:
    def test_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "evtexture_arch" in name:
                raise ImportError("No module named 'evtexture_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="EvTexture arch unavailable"):
                EvTextureRestorer._build_model(device)


class TestEvTextureArchModule:
    def test_evtexture_class_is_nn_module(self):
        from restorax.restorers.super_resolution.evtexture_arch import EvTexture

        model = EvTexture(scale=4)
        assert isinstance(model, torch.nn.Module)
