from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer


class TestScratchRemovalMeta:
    def test_name(self):
        assert ScratchRemovalRestorer().name == "scratch_removal"

    def test_category(self):
        assert ScratchRemovalRestorer().capabilities.category == RestorerCategory.ARTIFACT_REMOVAL

    def test_requires_temporal(self):
        assert ScratchRemovalRestorer().capabilities.requires_temporal is True

    def test_scale_factor(self):
        assert ScratchRemovalRestorer().capabilities.scale_factor == 1


class TestScratchRemovalLoadRaisesWhenArchAbsent:
    def test_raises_restorer_load_error_on_missing_propainter(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "propainter" in name.lower():
                raise ImportError("No module named 'propainter_arch'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError):
                ScratchRemovalRestorer().load(torch.device("cpu"))
