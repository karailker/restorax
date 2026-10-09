"""Tests for ModelRegistry LRU eviction logic."""

from __future__ import annotations

import torch

from restorax.core.exceptions import RestorerNotFoundError
from restorax.core.registry import ModelRegistry
from tests.conftest import IdentityRestorer


def _make_registry(max_loaded: int = 2) -> tuple[ModelRegistry, list[IdentityRestorer]]:
    """Return a registry with two pre-registered IdentityRestorer classes."""
    r1 = IdentityRestorer(scale=1)
    r2 = IdentityRestorer(scale=2)
    r3 = IdentityRestorer(scale=4)

    registry = ModelRegistry(max_loaded=max_loaded)
    registry._catalog["r1"] = lambda: r1  # type: ignore[assignment]
    registry._catalog["r2"] = lambda: r2  # type: ignore[assignment]
    registry._catalog["r3"] = lambda: r3  # type: ignore[assignment]
    return registry, [r1, r2, r3]


_CPU = torch.device("cpu")


def test_get_loads_model() -> None:
    registry, (r1, *_) = _make_registry()
    result = registry.get("r1", _CPU)
    assert result is r1
    assert r1.load_call_count == 1
    assert r1.is_loaded


def test_get_same_model_twice_does_not_reload() -> None:
    registry, (r1, *_) = _make_registry()
    registry.get("r1", _CPU)
    registry.get("r1", _CPU)
    assert r1.load_call_count == 1


def test_lru_eviction_when_cache_full() -> None:
    registry, (r1, r2, r3) = _make_registry(max_loaded=2)
    registry.get("r1", _CPU)
    registry.get("r2", _CPU)
    # r1 is LRU — requesting r3 should evict r1
    registry.get("r3", _CPU)

    assert r1.unload_call_count == 1
    assert not r1.is_loaded
    assert r2.is_loaded
    assert r3.is_loaded


def test_lru_access_updates_recency() -> None:
    registry, (r1, r2, r3) = _make_registry(max_loaded=2)
    registry.get("r1", _CPU)
    registry.get("r2", _CPU)
    # Touch r1 again — now r2 is the LRU
    registry.get("r1", _CPU)
    registry.get("r3", _CPU)  # should evict r2, not r1

    assert r2.unload_call_count == 1
    assert r1.is_loaded
    assert r3.is_loaded


def test_unknown_restorer_raises() -> None:
    registry = ModelRegistry()
    try:
        registry.get("nonexistent", _CPU)
        raise AssertionError("Should have raised")
    except RestorerNotFoundError:
        pass


def test_list_available() -> None:
    registry, _ = _make_registry()
    names = registry.list_available()
    assert "r1" in names
    assert "r2" in names


def test_unload_all() -> None:
    registry, (r1, r2, _) = _make_registry(max_loaded=2)
    registry.get("r1", _CPU)
    registry.get("r2", _CPU)
    registry.unload_all()
    assert r1.unload_call_count == 1
    assert r2.unload_call_count == 1
    assert registry.list_loaded() == []
