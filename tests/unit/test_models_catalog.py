from restorax.models_catalog import CATALOG, CATALOG_BY_NAME


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


class TestModelStatus:
    def test_every_entry_has_a_valid_status(self):
        from typing import get_args

        from restorax.models_catalog import CATALOG, Status

        valid = set(get_args(Status))
        assert all(m.status in valid for m in CATALOG)

    def test_status_table_has_no_stale_names(self):
        from restorax.models_catalog import _STATUS, CATALOG_BY_NAME

        assert set(_STATUS) <= set(CATALOG_BY_NAME)

    def test_every_non_ready_model_explains_why(self):
        from restorax.models_catalog import CATALOG

        for m in CATALOG:
            if m.status != "ready":
                assert len(m.note) > 30, f"{m.name} ({m.status}) needs an evidence note"

    def test_ready_models_have_a_weight_source(self):
        from restorax.models_catalog import CATALOG

        for m in CATALOG:
            if m.status == "ready" and m.group != "audio":
                assert m.urls or m.hf_repo, f"{m.name} is ready but has no weight source"

    def test_unavailable_models_are_skipped_by_download(self):
        from click.testing import CliRunner

        from restorax.cli_download import download_models_group

        result = CliRunner().invoke(download_models_group, ["--model", "tdm"])
        assert result.exit_code == 0
        assert "skipped tdm" in result.output
