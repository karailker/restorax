"""Unit tests for GaVSRestorer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from restorax.core.restorer import RestorerCategory
from restorax.restorers.stabilization.gavs import GaVSRestorer

torch = pytest.importorskip("torch")


class TestGaVSRestorerMeta:
    def test_name(self):
        assert GaVSRestorer().name == "gavs"

    def test_capabilities_category(self):
        assert GaVSRestorer().capabilities.category == RestorerCategory.STABILIZATION

    def test_capabilities_requires_temporal(self):
        assert GaVSRestorer().capabilities.requires_temporal is True


class TestGaVSRestorerLoad:
    def test_load_succeeds_with_opencv_fallback_and_logs_warning(self, caplog):
        """load() succeeds via the OpenCV fallback when gavs_arch is absent."""
        import logging

        mock_fallback = MagicMock()

        with (
            patch(
                "restorax.restorers.stabilization.gavs.GaVSRestorer._try_load_gavs",
                return_value=False,
            ),
            patch(
                "restorax.restorers.stabilization.gavs.GaVSRestorer._load_fallback",
                side_effect=lambda device: setattr(mock_fallback, "_loaded", True),
            ),
            caplog.at_level(logging.WARNING, logger="restorax.restorers.stabilization.gavs"),
        ):
            restorer = GaVSRestorer()
            restorer.load(torch.device("cpu"))

        assert restorer._loaded is True
        assert restorer._using_gavs is False

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("GAVS arch not yet publicly released" in m for m in warning_messages)
        assert any("https://arxiv.org/abs/2407.06009" in m for m in warning_messages)
