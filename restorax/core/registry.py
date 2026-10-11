from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from restorax.core.exceptions import RestorerNotFoundError
from restorax.core.restorer import BaseRestorer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Discovers and manages loaded restorer instances with an LRU VRAM cache.

    One registry instance lives per Celery worker process. When a pipeline
    stage requests a model that is not yet loaded, the registry loads it.
    If the loaded-model count would exceed max_loaded, the least-recently-used
    model is evicted (unloaded) first.

    This prevents OOM when multi-stage pipelines use more models than
    can simultaneously fit in VRAM (e.g., Real-ESRGAN + CodeFormer on 16 GB).
    """

    def __init__(self, max_loaded: int = 2) -> None:
        self._catalog: dict[str, type[BaseRestorer]] = {}
        self._loaded: OrderedDict[str, BaseRestorer] = OrderedDict()
        self._max_loaded = max_loaded

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, cls: type[BaseRestorer]) -> None:
        """Register a restorer class. The class must implement BaseRestorer."""
        # Instantiate temporarily to read the name property
        instance = object.__new__(cls)
        name = cls.name.fget(instance)  # type: ignore[attr-defined]
        self._catalog[name] = cls
        logger.debug("Registered restorer: %s", name)

    def register_all(self, classes: list[type[BaseRestorer]]) -> None:
        for cls in classes:
            self.register(cls)

    # ── Lookup / loading ──────────────────────────────────────────────────────

    def get(self, name: str, device: torch.device) -> BaseRestorer:
        """
        Return a loaded restorer instance, loading it on first access.
        Evicts the LRU model if the cache is at capacity.
        """
        if name in self._loaded:
            # Move to end (most recently used)
            self._loaded.move_to_end(name)
            return self._loaded[name]

        if name not in self._catalog:
            raise RestorerNotFoundError(
                f"Restorer '{name}' is not registered. Available: {sorted(self._catalog)}"
            )

        self._evict_if_needed()

        logger.info("Loading restorer '%s' on %s", name, device)
        instance = self._catalog[name]()
        instance.load(device)
        self._loaded[name] = instance
        return instance

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_available(self) -> list[str]:
        """Names of all registered restorers (loaded or not)."""
        return sorted(self._catalog)

    def list_loaded(self) -> list[str]:
        return list(self._loaded)

    def capabilities(self, name: str) -> object:
        """Return capabilities of a registered restorer without loading it."""
        if name not in self._catalog:
            raise RestorerNotFoundError(name)
        # Instantiate without loading to inspect capabilities
        instance = object.__new__(self._catalog[name])
        return self._catalog[name].capabilities.fget(instance)  # type: ignore[attr-defined]

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def unload_all(self) -> None:
        for restorer in self._loaded.values():
            restorer.unload()
        self._loaded.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        if len(self._loaded) >= self._max_loaded:
            evicted_name, evicted = self._loaded.popitem(last=False)
            logger.info("Evicting model '%s' from VRAM (LRU)", evicted_name)
            evicted.unload()
