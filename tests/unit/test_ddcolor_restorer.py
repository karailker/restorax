"""Unit tests for DDColorRestorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.colorization.ddcolor import DDColorRestorer

torch = pytest.importorskip("torch")


class TestDDColorRestorerMeta:
    def test_name(self):
        assert DDColorRestorer().name == "ddcolor"

    def test_capabilities_category(self):
        assert DDColorRestorer().capabilities.category == RestorerCategory.COLORIZATION

    def test_capabilities_scale_factor(self):
        assert DDColorRestorer().capabilities.scale_factor == 1

    def test_capabilities_requires_temporal(self):
        assert DDColorRestorer().capabilities.requires_temporal is False

    def test_capabilities_color_spaces(self):
        caps = DDColorRestorer().capabilities
        assert caps.input_color_space == "rgb"
        assert caps.output_color_space == "rgb"


class TestDDColorVariants:
    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="variant"):
            DDColorRestorer(variant="nope")

    def test_model_kwargs_follow_the_variant(self):
        from restorax.restorers.colorization.ddcolor import _model_kwargs

        assert _model_kwargs("paper_tiny", None)["encoder_name"] == "convnext-t"
        assert _model_kwargs("modelscope", None)["encoder_name"] == "convnext-l"
        assert _model_kwargs("artistic", None)["num_output_channels"] == 2

    def test_config_json_overrides_known_keys_only(self):
        from restorax.restorers.colorization.ddcolor import _model_kwargs

        kwargs = _model_kwargs("modelscope", {"num_queries": 256, "bogus": 1})
        assert kwargs["num_queries"] == 256
        assert "bogus" not in kwargs


class _FakeEntryNotFoundError(Exception):
    """Stands in for huggingface_hub.errors.EntryNotFoundError."""


def _install_fake_hub(monkeypatch, tmp_path, download):
    """Install a fake ``huggingface_hub`` (the real one may be stubbed by other test modules)."""
    import sys
    import types
    from types import SimpleNamespace

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download  # type: ignore[attr-defined]
    errors = types.ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = _FakeEntryNotFoundError  # type: ignore[attr-defined]
    hub.errors = errors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    # Replace the module attribute itself: other tests may have swapped the settings object.
    monkeypatch.setattr("restorax.config.settings", SimpleNamespace(model_dir=str(tmp_path)))


class TestDDColorWeights:
    def test_prefers_safetensors_then_falls_back(self, monkeypatch, tmp_path):
        from restorax.restorers.colorization import ddcolor

        asked: list[str] = []

        def fake_download(repo_id, filename, local_dir):
            asked.append(filename)
            if filename in ("config.json", "model.safetensors"):
                raise _FakeEntryNotFoundError("missing")
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
            return str(target)

        _install_fake_hub(monkeypatch, tmp_path, fake_download)
        path, config = ddcolor._download_weights("paper_tiny")
        assert path.name == "pytorch_model.bin" and config is None
        assert asked == ["config.json", "model.safetensors", "pytorch_model.bin"]

    def test_no_weight_file_is_a_load_error(self, monkeypatch, tmp_path):
        from restorax.restorers.colorization import ddcolor

        def fake_download(repo_id, filename, local_dir):
            raise _FakeEntryNotFoundError("missing")

        _install_fake_hub(monkeypatch, tmp_path, fake_download)
        with pytest.raises(RestorerLoadError, match="No weight file"):
            ddcolor._download_weights("modelscope")


class TestDDColorBuildModel:
    """Architecture + checkpoint handling with a randomly initialised DDColor-T (no network)."""

    @staticmethod
    def _patch_weights(monkeypatch, tmp_path, state_dict):
        from restorax.restorers.colorization import ddcolor

        path = tmp_path / "pytorch_model.pt"
        torch.save(state_dict, path)
        monkeypatch.setattr(ddcolor, "_download_weights", lambda variant: (path, None))

    def test_loads_a_matching_checkpoint(self, monkeypatch, tmp_path):
        from restorax.restorers.colorization.ddcolor import _model_kwargs
        from restorax.restorers.colorization.ddcolor_arch import DDColor

        reference = DDColor(**_model_kwargs("paper_tiny", None))
        self._patch_weights(monkeypatch, tmp_path, {"params": reference.state_dict()})
        model = DDColorRestorer._build_model("paper_tiny", torch.device("cpu"))
        assert isinstance(model, DDColor)

    def test_mismatching_checkpoint_is_rejected(self, monkeypatch, tmp_path):
        self._patch_weights(monkeypatch, tmp_path, {"some.other.weight": torch.zeros(1)})
        with pytest.raises(RestorerLoadError, match="does not match"):
            DDColorRestorer._build_model("paper_tiny", torch.device("cpu"))


class TestDDColorPreservesLuminance:
    def test_output_keeps_the_source_l_channel(self):
        import cv2
        import numpy as np

        from restorax.core.restorer import RestorerParams

        r = DDColorRestorer()
        r._model = lambda x: torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])  # type: ignore[assignment]
        r._device = torch.device("cpu")
        r._loaded = True
        frame = np.random.default_rng(1).integers(0, 255, (48, 80, 3), dtype=np.uint8)
        out = r.process_frame(frame, RestorerParams())
        assert out.shape == frame.shape and out.dtype == np.uint8
        l_in = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 0].astype(int)
        l_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[:, :, 0].astype(int)
        assert np.abs(l_in - l_out).mean() < 1.5
