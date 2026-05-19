"""Integration tests for `restorax download-models` CLI command."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Set env vars before any restorax import so pydantic_settings resolves them.
os.environ.setdefault("RESTORAX_DATABASE_URL", "sqlite+aiosqlite:///./test_restorax.db")
os.environ.setdefault("RESTORAX_REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("RESTORAX_DEVICE", "cpu")
os.environ.setdefault("RESTORAX_STORAGE_LOCAL_ROOT", "/tmp/restorax_test_data")

# Stub torch so the root conftest (and any transitive imports) don't fail.
if "torch" not in sys.modules:
    _torch_stub = MagicMock()
    sys.modules["torch"] = _torch_stub
    sys.modules["torch.nn"] = _torch_stub
    sys.modules["torch.cuda"] = _torch_stub

# Stub huggingface_hub — not installed in the test env.
if "huggingface_hub" not in sys.modules:
    _hfh_stub = MagicMock()
    sys.modules["huggingface_hub"] = _hfh_stub

import huggingface_hub as _hfh  # noqa: E402  (now guaranteed to be the stub)

# Stub pydantic_settings — not installed in the test env.
# restorax.config does:  from pydantic_settings import BaseSettings, SettingsConfigDict
# We need a real class for BaseSettings (subclassed by Settings), and a dummy
# SettingsConfigDict.  Use MagicMock for the module so all attribute accesses succeed.
if "pydantic_settings" not in sys.modules:
    import types

    _ps = MagicMock()

    class _BaseSettings:
        """Minimal stand-in that lets Settings() be instantiated."""
        model_config: dict = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

        def __init__(self, **kwargs):
            # Apply defaults from class-level field annotations, then kwargs.
            import inspect
            for name, annotation in inspect.get_annotations(type(self), eval_str=False).items():
                if hasattr(type(self), name):
                    setattr(self, name, getattr(type(self), name))
            for k, v in kwargs.items():
                setattr(self, k, v)

    _ps.BaseSettings = _BaseSettings
    _ps.SettingsConfigDict = dict  # close enough — it's just called to build a dict
    sys.modules["pydantic_settings"] = _ps

from restorax.cli_download import download_models_group  # noqa: E402
from restorax.models_catalog import ModelEntry  # noqa: E402

import click  # noqa: E402


# Thin wrapper group so we can invoke `download-models` via CliRunner the same
# way as through the real `cli` group — without importing cli.py (which pulls
# in torch-dependent modules).
@click.group()
def _test_cli() -> None:
    pass


_test_cli.add_command(download_models_group)


# ---------------------------------------------------------------------------
# Minimal fake catalog used across tests
# ---------------------------------------------------------------------------
_FAKE_CATALOG: list[ModelEntry] = [
    ModelEntry("fake_sr", "sr", "org/fake-sr", ["model.pth"], 50),
    ModelEntry("fake_face", "face", "org/fake-face", ["face.pth"], 100),
    ModelEntry("fake_snap", "diffusion", "org/fake-snap", [], 500, snapshot=True),
]
_FAKE_CATALOG_BY_NAME = {m.name: m for m in _FAKE_CATALOG}


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _patch_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real catalog with the fake one in cli_download's module scope."""
    monkeypatch.setattr("restorax.models_catalog.CATALOG", _FAKE_CATALOG)
    monkeypatch.setattr("restorax.models_catalog.CATALOG_BY_NAME", _FAKE_CATALOG_BY_NAME)


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

class TestHelp:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(_test_cli, ["download-models", "--help"])
        assert result.exit_code == 0

    def test_help_lists_options(self, runner: CliRunner) -> None:
        result = runner.invoke(_test_cli, ["download-models", "--help"])
        assert "--model" in result.output
        assert "--group" in result.output
        assert "--all" in result.output
        assert "--force" in result.output


# ---------------------------------------------------------------------------
# No-args → status table
# ---------------------------------------------------------------------------

class TestStatusTable:
    def test_no_args_prints_table(self, runner: CliRunner) -> None:
        result = runner.invoke(_test_cli, ["download-models"])
        assert result.exit_code == 0
        # Table title
        assert "Model Weights Status" in result.output

    def test_status_table_contains_model_names(self, runner: CliRunner) -> None:
        result = runner.invoke(_test_cli, ["download-models"])
        assert "fake_sr" in result.output
        assert "fake_face" in result.output
        assert "fake_snap" in result.output


# ---------------------------------------------------------------------------
# --model unknown → warning, exit 0
# ---------------------------------------------------------------------------

class TestUnknownModel:
    def test_unknown_model_prints_warning(self, runner: CliRunner) -> None:
        result = runner.invoke(_test_cli, ["download-models", "--model", "does_not_exist"])
        assert "Warning" in result.output or "unknown model" in result.output.lower()

    def test_unknown_model_exits_zero(self, runner: CliRunner) -> None:
        # No valid model selected after warning → "No models selected." and exit 0
        result = runner.invoke(_test_cli, ["download-models", "--model", "does_not_exist"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --model <known> → calls hf_hub_download
# ---------------------------------------------------------------------------

class TestDownloadSingleModel:
    def test_calls_hf_hub_download(self, runner: CliRunner, tmp_path) -> None:
        # hf_hub_download / snapshot_download are imported lazily inside the command,
        # so patch them on the huggingface_hub module itself.
        with (
            patch.object(_hfh, "hf_hub_download") as mock_dl,
            patch.object(_hfh, "snapshot_download"),
            patch.object(ModelEntry, "is_ready", return_value=False),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--model", "fake_sr"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            repo_id="org/fake-sr",
            filename="model.pth",
            local_dir=str(tmp_path),
        )
        assert "✓ fake_sr" in result.output

    def test_calls_snapshot_download_for_snapshot_model(self, runner: CliRunner, tmp_path) -> None:
        with (
            patch.object(_hfh, "hf_hub_download"),
            patch.object(_hfh, "snapshot_download") as mock_snap,
            patch.object(ModelEntry, "is_ready", return_value=False),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--model", "fake_snap"])
        assert result.exit_code == 0
        mock_snap.assert_called_once_with(
            repo_id="org/fake-snap",
            local_dir=str(tmp_path),
        )
        assert "✓ fake_snap" in result.output


# ---------------------------------------------------------------------------
# --group → downloads all models in that group
# ---------------------------------------------------------------------------

class TestDownloadGroup:
    def test_group_sr_downloads_sr_models(self, runner: CliRunner, tmp_path) -> None:
        with (
            patch.object(_hfh, "hf_hub_download") as mock_dl,
            patch.object(_hfh, "snapshot_download"),
            patch.object(ModelEntry, "is_ready", return_value=False),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--group", "sr"])
        assert result.exit_code == 0
        # fake_sr is in group sr; fake_face is not
        assert "✓ fake_sr" in result.output
        assert "fake_face" not in result.output or "✓ fake_face" not in result.output


# ---------------------------------------------------------------------------
# --all → downloads everything
# ---------------------------------------------------------------------------

class TestDownloadAll:
    def test_all_flag_downloads_all_models(self, runner: CliRunner, tmp_path) -> None:
        with (
            patch.object(_hfh, "hf_hub_download"),
            patch.object(_hfh, "snapshot_download"),
            patch.object(ModelEntry, "is_ready", return_value=False),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--all"])
        assert result.exit_code == 0
        assert "✓ fake_sr" in result.output
        assert "✓ fake_face" in result.output
        assert "✓ fake_snap" in result.output


# ---------------------------------------------------------------------------
# Already-ready model is skipped without --force
# ---------------------------------------------------------------------------

class TestSkipReady:
    def test_skips_ready_model_without_force(self, runner: CliRunner, tmp_path) -> None:
        with (
            patch.object(_hfh, "hf_hub_download") as mock_dl,
            patch.object(_hfh, "snapshot_download") as mock_snap,
            patch.object(ModelEntry, "is_ready", return_value=True),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--model", "fake_sr"])
        assert result.exit_code == 0
        mock_dl.assert_not_called()
        mock_snap.assert_not_called()
        assert "skipped" in result.output

    def test_force_redownloads_ready_model(self, runner: CliRunner, tmp_path) -> None:
        with (
            patch.object(_hfh, "hf_hub_download") as mock_dl,
            patch.object(_hfh, "snapshot_download"),
            patch.object(ModelEntry, "is_ready", return_value=True),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--model", "fake_sr", "--force"])
        assert result.exit_code == 0
        mock_dl.assert_called_once()


# ---------------------------------------------------------------------------
# huggingface_hub missing → error + exit 1
# ---------------------------------------------------------------------------

class TestMissingHuggingfaceHub:
    def test_missing_hf_hub_exits_nonzero(self, runner: CliRunner) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = runner.invoke(_test_cli, ["download-models", "--model", "fake_sr"])
        # exit_code 1 or SystemExit(1) is captured by CliRunner
        assert result.exit_code != 0

    def test_missing_hf_hub_prints_install_hint(self, runner: CliRunner) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = runner.invoke(_test_cli, ["download-models", "--model", "fake_sr"])
        assert "huggingface_hub" in result.output


# ---------------------------------------------------------------------------
# Download failure → warning, continues, exit 0
# ---------------------------------------------------------------------------

class TestDownloadFailure:
    def test_failure_prints_warning_and_continues(self, runner: CliRunner, tmp_path) -> None:
        with (
            patch.object(_hfh, "hf_hub_download", side_effect=RuntimeError("network error")),
            patch.object(_hfh, "snapshot_download"),
            patch.object(ModelEntry, "is_ready", return_value=False),
            patch.object(ModelEntry, "weight_dir", return_value=tmp_path),
        ):
            result = runner.invoke(_test_cli, ["download-models", "--all"])
        assert result.exit_code == 0
        assert "Warning" in result.output
        assert "network error" in result.output
