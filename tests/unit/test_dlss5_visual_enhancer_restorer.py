"""Unit tests for the external-CLI DLSS 5 Visual Enhancer adapter (no GPU / Windows needed)."""

# ruff: noqa: E402  (imports follow pytest.importorskip)
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory, RestorerParams

torch = pytest.importorskip("torch")

from restorax.restorers.enhancement import dlss5_visual_enhancer as mod
from restorax.restorers.enhancement.dlss5_visual_enhancer import (
    DLSS5VisualEnhancerRestorer,
)


def _frame(h: int = 8, w: int = 12) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _loaded(monkeypatch, tmp_path: Path, platform: str = "win32") -> DLSS5VisualEnhancerRestorer:
    cli = tmp_path / "VE_CLI.exe"
    cli.write_bytes(b"")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("RESTORAX_VE_CLI", str(cli))
    r = DLSS5VisualEnhancerRestorer()
    r.load(torch.device("cpu"))
    return r


def _fake_cli(monkeypatch, *, transform=lambda img: img, returncode=0, stdout="", stderr=""):
    """Replace subprocess.run: the 'CLI' reads the input PNG and writes transform(img) to -o."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if returncode == 0:
            img = cv2.imread(cmd[cmd.index("render") + 1], cv2.IMREAD_COLOR)
            cv2.imwrite(cmd[cmd.index("-o") + 1], transform(img))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    return calls


class TestMeta:
    def test_name_and_category(self):
        r = DLSS5VisualEnhancerRestorer()
        assert r.name == "dlss5_visual_enhancer"
        assert r.capabilities.category == RestorerCategory.ENHANCEMENT
        assert r.capabilities.scale_factor == 1

    def test_param_schema_names(self):
        assert [s.name for s in DLSS5VisualEnhancerRestorer.PARAM_SCHEMA] == ["style", "strength"]


class TestLoad:
    def test_linux_without_wine_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("RESTORAX_VE_LAUNCHER", raising=False)
        monkeypatch.setattr(mod.shutil, "which", lambda _: None)
        with pytest.raises(RestorerLoadError, match="Wine"):
            DLSS5VisualEnhancerRestorer().load(torch.device("cpu"))

    def test_linux_defaults_to_wine(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RESTORAX_VE_LAUNCHER", raising=False)
        real_which = mod.shutil.which
        monkeypatch.setattr(
            mod.shutil, "which", lambda n: "/usr/bin/wine" if n == "wine" else real_which(n)
        )
        r = _loaded(monkeypatch, tmp_path, platform="linux")
        assert r._launcher == ["wine"]

    def test_launcher_env_is_split(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RESTORAX_VE_LAUNCHER", "env WINEPREFIX=/p wine")
        r = _loaded(monkeypatch, tmp_path, platform="linux")
        assert r._launcher == ["env", "WINEPREFIX=/p", "wine"]

    def test_windows_has_no_launcher(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RESTORAX_VE_LAUNCHER", "wine")
        assert _loaded(monkeypatch, tmp_path)._launcher == []

    def test_missing_cli_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("RESTORAX_VE_CLI", raising=False)
        monkeypatch.setattr(mod.shutil, "which", lambda _: None)
        with pytest.raises(RestorerLoadError, match="RESTORAX_VE_CLI"):
            DLSS5VisualEnhancerRestorer().load(torch.device("cpu"))

    def test_bad_path_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("RESTORAX_VE_CLI", str(tmp_path / "nope.exe"))
        with pytest.raises(RestorerLoadError, match="not found at"):
            DLSS5VisualEnhancerRestorer().load(torch.device("cpu"))

    def test_load_unload(self, monkeypatch, tmp_path):
        r = _loaded(monkeypatch, tmp_path)
        assert r.is_loaded
        r.unload()
        assert not r.is_loaded


class TestProcess:
    def test_requires_load(self):
        with pytest.raises(RuntimeError, match="load"):
            DLSS5VisualEnhancerRestorer().process_frame(_frame(), RestorerParams())

    def test_roundtrip_preserves_rgb_order(self, monkeypatch, tmp_path):
        r = _loaded(monkeypatch, tmp_path)
        _fake_cli(monkeypatch)
        frame = _frame()
        np.testing.assert_array_equal(r.process_frame(frame, RestorerParams()), frame)

    def test_cli_arguments(self, monkeypatch, tmp_path):
        r = _loaded(monkeypatch, tmp_path)
        calls = _fake_cli(monkeypatch)
        r.process_frame(_frame(), RestorerParams(extra={"style": "2", "strength": 1.5, "gpu": 0}))
        cmd = calls[0]
        assert cmd[1] == "render"
        assert cmd[cmd.index("--style") + 1] == "2"
        assert cmd[cmd.index("--strength") + 1] == "1.5"
        assert cmd[cmd.index("--gpu") + 1] == "0"

    @pytest.mark.parametrize("extra", [{"style": "9"}, {"strength": 3.0}, {"strength": -0.1}])
    def test_invalid_params(self, monkeypatch, tmp_path, extra):
        r = _loaded(monkeypatch, tmp_path)
        _fake_cli(monkeypatch)
        with pytest.raises(ValueError):
            r.process_frame(_frame(), RestorerParams(extra=extra))

    def test_linux_command_uses_launcher_and_winepath(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RESTORAX_VE_LAUNCHER", "wine")
        r = _loaded(monkeypatch, tmp_path, platform="linux")
        monkeypatch.setattr(
            mod.shutil, "which", lambda n: "/usr/bin/winepath" if n == "winepath" else None
        )
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[0] == "/usr/bin/winepath":
                return subprocess.CompletedProcess(
                    cmd, 0, "Z:" + cmd[2].replace("/", "\\") + "\n", ""
                )
            real_in = Path(cmd[cmd.index("render") + 1][2:].replace("\\", "/"))
            real_out = Path(cmd[cmd.index("-o") + 1][2:].replace("\\", "/"))
            cv2.imwrite(str(real_out), cv2.imread(str(real_in)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        r.process_frame(_frame(), RestorerParams())
        render = next(c for c in calls if c[0] == "wine")
        assert render[1].endswith("VE_CLI.exe")
        assert render[render.index("render") + 1].startswith("Z:")

    def test_cli_failure_surfaces_json_error(self, monkeypatch, tmp_path):
        r = _loaded(monkeypatch, tmp_path)
        _fake_cli(
            monkeypatch, returncode=1, stdout=json.dumps({"ok": False, "error": "no RTX GPU"})
        )
        with pytest.raises(RuntimeError, match="no RTX GPU"):
            r.process_frame(_frame(), RestorerParams())

    def test_size_change_rejected(self, monkeypatch, tmp_path):
        r = _loaded(monkeypatch, tmp_path)
        _fake_cli(monkeypatch, transform=lambda img: cv2.resize(img, None, fx=2, fy=2))
        with pytest.raises(RuntimeError, match="changed frame size"):
            r.process_frame(_frame(), RestorerParams())

    def test_timeout(self, monkeypatch, tmp_path):
        r = _loaded(monkeypatch, tmp_path)

        def boom(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])

        monkeypatch.setattr(mod.subprocess, "run", boom)
        with pytest.raises(RuntimeError, match="timed out"):
            r.process_frame(_frame(), RestorerParams(extra={"timeout": 1}))
