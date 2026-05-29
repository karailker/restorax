from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.frame_interpolation.rife import RIFERestorer


class TestRIFEMeta:
    def test_name(self):
        assert RIFERestorer().name == "rife_v4"

    def test_category(self):
        assert RIFERestorer().capabilities.category == RestorerCategory.FRAME_INTERPOLATION

    def test_requires_temporal(self):
        assert RIFERestorer().capabilities.requires_temporal is True


class TestRIFELoadRaisesWhenArchAbsent:
    def test_raises_restorer_load_error_on_missing_rife_arch(self, tmp_path):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "rife_arch" in name:
                raise ImportError("No module named 'rife_arch'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("restorax.config.settings.model_dir", str(tmp_path)),
        ):
            with pytest.raises(RestorerLoadError, match="RIFE arch unavailable"):
                RIFERestorer().load(torch.device("cpu"))
