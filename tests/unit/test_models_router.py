"""Tests for GET /models — verifies all restorer categories are present."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from restorax.api.app import app
    return TestClient(app)


def test_models_includes_audio_restorers(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    names = {r["name"] for r in resp.json()["restorers"]}
    assert "demucs_htdemucs" in names, f"demucs_htdemucs missing — got {names}"
    assert "voicefixer" in names, f"voicefixer missing — got {names}"
    assert "rnnoise" in names, f"rnnoise missing — got {names}"


def test_models_lists_all_categories(client):
    resp = client.get("/models")
    categories = {r["category"] for r in resp.json()["restorers"]}
    expected = {
        "super_resolution", "face_restoration", "colorization",
        "frame_interpolation", "artifact_removal", "hdr_conversion",
        "stabilization", "deinterlacing",
        "source_separation", "speech_enhancement", "noise_suppression",
    }
    assert expected <= categories, f"Missing categories: {expected - categories}"


def test_models_response_fields(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    audio_categories = {"source_separation", "speech_enhancement", "noise_suppression"}
    for r in resp.json()["restorers"]:
        assert "name" in r
        assert "category" in r
        assert "tags" in r
        if r["category"] in audio_categories:
            assert r.get("min_ram_gb") is not None, f"audio restorer {r['name']} missing min_ram_gb"
            assert r.get("supports_stereo") is not None, f"audio restorer {r['name']} missing supports_stereo"
        else:
            assert r.get("input_color_space") is not None, f"video restorer {r['name']} missing input_color_space"
            assert r.get("min_vram_gb") is not None, f"video restorer {r['name']} missing min_vram_gb"
