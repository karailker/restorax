from pathlib import Path
import pytest
from restorax.models_catalog import CATALOG, CATALOG_BY_NAME, ModelEntry


def test_catalog_has_entries():
    assert len(CATALOG) >= 19


def test_catalog_by_name_lookup():
    entry = CATALOG_BY_NAME["real_esrgan"]
    assert entry.hf_repo == "xinntao/Real-ESRGAN"
    assert entry.weight_files == ["RealESRGANx4plus.pth"]
    assert entry.size_mb == 67
    assert entry.group == "sr"


def test_diffusion_models_use_snapshot():
    for name in ("seedvr", "tdm", "upscale_a_video"):
        assert CATALOG_BY_NAME[name].snapshot is True


def test_is_ready_false_for_nonexistent(tmp_path, monkeypatch):
    from restorax.config import settings
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    entry = CATALOG_BY_NAME["real_esrgan"]
    assert entry.is_ready() is False


def test_all_entries_have_required_fields():
    for entry in CATALOG:
        assert entry.name
        assert entry.group in ("sr", "face", "diffusion", "extras", "audio")
        assert entry.hf_repo
        assert entry.size_mb >= 0
