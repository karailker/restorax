"""
Plugin discovery for third-party RestoraX restorers.

Third-party packages register restorer classes under the
`restorax.restorers` importlib.metadata entry-points group:

    # In plugin's pyproject.toml:
    [project.entry-points."restorax.restorers"]
    my_restorer = "my_package.restorer:MyRestorerClass"

At runtime, `discover_plugins()` loads all registered classes and
returns them for registration with a `ModelRegistry` instance.

This allows the ecosystem to extend RestoraX without modifying the core
package — a plugin only needs to install itself and the new restorer
appears automatically in the registry, CLI model list, and API.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from restorax.core.restorer import BaseRestorer

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "restorax.restorers"


def discover_plugins() -> list[type[BaseRestorer]]:
    """
    Load all restorer classes registered under `restorax.restorers`.

    Returns:
        List of BaseRestorer subclasses discovered from installed packages.
        Classes that fail to load are skipped with a warning.
    """
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    classes: list[type[BaseRestorer]] = []

    for ep in eps:
        try:
            cls = ep.load()
            if not (isinstance(cls, type) and issubclass(cls, BaseRestorer)):
                logger.warning(
                    "Plugin entry point '%s' loaded '%s' which is not a BaseRestorer subclass — skipped",
                    ep.name,
                    cls,
                )
                continue
            classes.append(cls)
            logger.debug("Loaded plugin restorer: %s (from %s)", ep.name, ep.value)
        except Exception as exc:
            logger.warning("Failed to load plugin '%s': %s", ep.name, exc)

    return classes


def register_plugins(registry: object) -> int:
    """
    Discover all plugins and register them with a ModelRegistry.

    Args:
        registry: A `ModelRegistry` instance.

    Returns:
        Number of successfully registered plugin restorers.
    """
    from restorax.core.registry import ModelRegistry

    assert isinstance(registry, ModelRegistry)
    classes = discover_plugins()
    registered = 0

    for cls in classes:
        try:
            registry.register(cls)
            registered += 1
        except Exception as exc:
            logger.warning("Could not register plugin class %s: %s", cls, exc)

    if registered:
        logger.info("Registered %d plugin restorer(s) from entry points", registered)

    return registered
