"""Tests for plugin discovery and registration."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from restorax.core.plugin import discover_plugins, register_plugins
from restorax.core.registry import ModelRegistry
from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

# ── Fake plugin restorer ──────────────────────────────────────────────────────


class _FakePluginRestorer(BaseRestorer):
    @property
    def name(self) -> str:
        return "fake_plugin_restorer"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            scale_factor=2,
        )

    def load(self, device: torch.device) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        return frame


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_fake_entry_point(cls: type) -> MagicMock:
    ep = MagicMock(spec=importlib.metadata.EntryPoint)
    ep.name = "fake_plugin_restorer"
    ep.value = f"{cls.__module__}:{cls.__qualname__}"
    ep.load.return_value = cls
    return ep


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_discover_plugins_returns_valid_classes() -> None:
    ep = _make_fake_entry_point(_FakePluginRestorer)
    with patch("restorax.core.plugin.entry_points", return_value=[ep]):
        classes = discover_plugins()
    assert _FakePluginRestorer in classes


def test_discover_plugins_skips_non_base_restorer() -> None:
    """Entry points pointing to non-BaseRestorer classes must be skipped."""
    ep = MagicMock()
    ep.name = "bad_plugin"
    ep.load.return_value = str  # not a BaseRestorer
    with patch("restorax.core.plugin.entry_points", return_value=[ep]):
        classes = discover_plugins()
    assert str not in classes


def test_discover_plugins_skips_on_import_error() -> None:
    ep = MagicMock()
    ep.name = "broken_plugin"
    ep.load.side_effect = ImportError("missing dep")
    with patch("restorax.core.plugin.entry_points", return_value=[ep]):
        classes = discover_plugins()
    assert classes == []


def test_register_plugins_adds_to_registry() -> None:
    ep = _make_fake_entry_point(_FakePluginRestorer)
    registry = ModelRegistry(max_loaded=2)
    with patch("restorax.core.plugin.entry_points", return_value=[ep]):
        count = register_plugins(registry)
    assert count == 1
    assert "fake_plugin_restorer" in registry.list_available()


def test_register_plugins_no_plugins_returns_zero() -> None:
    registry = ModelRegistry(max_loaded=2)
    with patch("restorax.core.plugin.entry_points", return_value=[]):
        count = register_plugins(registry)
    assert count == 0


def test_built_in_restorers_discoverable_via_entry_points() -> None:
    """Verify built-in restorers are registered in pyproject.toml entry points."""
    eps = importlib.metadata.entry_points(group="restorax.restorers")
    names = {ep.name for ep in eps}
    assert "real_esrgan_x4plus" in names
    assert "codeformer" in names
    assert "rife_v4" in names
    assert "ddcolor" in names
