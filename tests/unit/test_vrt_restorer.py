"""Unit tests for VRTRestorer."""
from __future__ import annotations

import pytest

from restorax.restorers.super_resolution.vrt import VRTRestorer
from restorax.core.restorer import RestorerCategory

torch = pytest.importorskip("torch")


class TestVRTRestorerMeta:
    def test_name(self):
        r = VRTRestorer()
        assert r.name == "vrt_x4"

    def test_capabilities_category(self):
        caps = VRTRestorer().capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION

    def test_capabilities_scale_factor(self):
        caps = VRTRestorer().capabilities
        assert caps.scale_factor == 4

    def test_capabilities_requires_temporal(self):
        caps = VRTRestorer().capabilities
        assert caps.requires_temporal is True


# ponytail: VRT arch is vendored (vrt_arch.py, no basicsr dep), so "basicsr
# absent" can no longer trigger an arch-load failure — that test class was
# deleted. The real guard now is the no-weights RestorerLoadError in _build_model.
