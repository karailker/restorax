"""Unit tests for MambaIRRestorer."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer

torch = pytest.importorskip("torch")


class TestMambaIRRestorerMeta:
    def test_name(self):
        r = MambaIRRestorer()
        assert r.name == "mamba_ir_x4"

    def test_capabilities_category(self):
        caps = MambaIRRestorer().capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION

    def test_capabilities_scale_factor(self):
        caps = MambaIRRestorer().capabilities
        assert caps.scale_factor == 4


class TestMambaIRBuildModelRaisesWhenMambaSsmAbsent:
    def test_raises_restorer_load_error_on_import_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mamba_ssm" or name.startswith("mamba_ssm."):
                raise ImportError("No module named 'mamba_ssm'")
            if "mamba_ir_arch" in name:
                raise ImportError("No module named 'mamba_ir_arch'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="MambaIR unavailable"):
                MambaIRRestorer._build_model(device)

    def test_vendored_arch_module_imports_or_missing_deps_raise_load_error(self):
        """Real (unmocked) smoke test: the vendored mamba_ir_arch module exists
        and imports cleanly on its own; if the optional runtime deps (timm,
        mamba-ssm) aren't installed in this environment, _build_model must
        surface that as a RestorerLoadError rather than a raw ImportError."""
        try:
            from restorax.restorers.super_resolution.mamba_ir_arch import MambaIR

            assert MambaIR.__name__ == "MambaIR"
        except ImportError:
            device = torch.device("cpu")
            with pytest.raises(RestorerLoadError, match="MambaIR unavailable"):
                MambaIRRestorer._build_model(device)

    def test_raises_restorer_load_error_on_basicsr_absent(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "basicsr" or name.startswith("basicsr."):
                raise ImportError("No module named 'basicsr'")
            if "mamba_ir_arch" in name:
                raise ImportError("No module named 'basicsr'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="MambaIR unavailable"):
                MambaIRRestorer._build_model(device)
