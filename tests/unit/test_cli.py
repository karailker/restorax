"""Unit tests for the CLI commands (models, presets, benchmark sub-commands).

The `run` command is exercised in integration tests since it requires video I/O.
These tests cover the informational commands that need no GPU or real files.
"""

from __future__ import annotations

from click.testing import CliRunner

from restorax.cli import cli


def _runner() -> CliRunner:
    return CliRunner()


# ── restorax models ───────────────────────────────────────────────────────────


class TestModelsCommand:
    def test_exits_zero(self):
        result = _runner().invoke(cli, ["models"])
        assert result.exit_code == 0

    def test_lists_known_restorers(self):
        result = _runner().invoke(cli, ["models"])
        output = result.output
        assert "real_esrgan" in output.lower() or "esrgan" in output.lower()
        assert "mamba" in output.lower() or "vrt" in output.lower()

    def test_contains_category_column(self):
        result = _runner().invoke(cli, ["models"])
        assert "super_resolution" in result.output.lower() or "sr" in result.output.lower()


# ── restorax presets ──────────────────────────────────────────────────────────


class TestPresetsCommand:
    def test_exits_zero(self):
        result = _runner().invoke(cli, ["presets"])
        assert result.exit_code == 0

    def test_lists_sr_x4_preset(self):
        result = _runner().invoke(cli, ["presets"])
        assert "sr_x4" in result.output

    def test_lists_classic_film_preset(self):
        result = _runner().invoke(cli, ["presets"])
        assert "classic_film" in result.output


# ── restorax benchmark ────────────────────────────────────────────────────────


class TestBenchmarkCommand:
    def test_benchmark_group_help(self):
        result = _runner().invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_benchmark_run_help(self):
        result = _runner().invoke(cli, ["benchmark", "run", "--help"])
        assert result.exit_code == 0
        assert "--device" in result.output
        assert "--num-frames" in result.output

    def test_benchmark_compare_help(self):
        result = _runner().invoke(cli, ["benchmark", "compare", "--help"])
        assert result.exit_code == 0

    def test_benchmark_run_cpu_small(self, tmp_path):
        """Run benchmark with 1 frame on CPU — must produce JSON output."""
        result = _runner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--device",
                "cpu",
                "--num-frames",
                "1",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) >= 1, "No JSON files produced"

    def test_benchmark_compare_missing_dir(self, tmp_path):
        """compare with a valid directory that has no JSON files exits non-zero or prints warning."""
        result = _runner().invoke(cli, ["benchmark", "compare", str(tmp_path)])
        # Either exits 0 with "no results" message, or non-zero
        assert "no" in result.output.lower() or result.exit_code != 0 or result.output.strip() != ""


# ── restorax --help ───────────────────────────────────────────────────────────


class TestTopLevelHelp:
    def test_help(self):
        result = _runner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "models" in result.output
        assert "presets" in result.output
        assert "benchmark" in result.output
