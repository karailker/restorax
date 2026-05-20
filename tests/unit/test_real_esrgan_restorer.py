from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer


class TestRealESRGANMeta:
    def test_name(self):
        assert RealESRGANx4Restorer().name == "real_esrgan_x4plus"

    def test_category(self):
        assert RealESRGANx4Restorer().capabilities.category == RestorerCategory.SUPER_RESOLUTION

    def test_scale_factor(self):
        assert RealESRGANx4Restorer().capabilities.scale_factor == 4


class TestRealESRGANLoadRaisesWhenBasicsrAbsent:
    def test_raises_restorer_load_error_on_missing_basicsr(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "basicsr" or name.startswith("basicsr."):
                raise ImportError("No module named 'basicsr'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="basicsr"):
                RealESRGANx4Restorer().load(torch.device("cpu"))
