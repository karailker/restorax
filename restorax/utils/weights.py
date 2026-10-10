"""Download model weights from plain HTTPS URLs (e.g. GitHub release assets)."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from restorax.core.exceptions import RestorerLoadError

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20


def download_file(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    min_bytes: int = 1,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest`` atomically and return ``dest``.

    The payload is written to a temporary file next to ``dest`` and moved into
    place only after it is complete (and, when given, matches ``sha256`` and is
    at least ``min_bytes`` long), so a failed or truncated transfer never leaves
    a half-written weight file behind.

    Raises:
        RestorerLoadError: on network errors, truncated or corrupt downloads.
    """
    if not url.startswith("https://"):
        raise RestorerLoadError(f"Refusing to download weights over a non-HTTPS URL: {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    tmp = Path(tmp_name)
    digest = hashlib.sha256()
    total = 0
    try:
        logger.info("Downloading %s -> %s", url, dest)
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310
            os.fdopen(fd, "wb") as out,
        ):
            expected = response.headers.get("Content-Length")
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        if expected is not None and total != int(expected):
            raise RestorerLoadError(
                f"Truncated download from {url}: got {total} of {expected} bytes"
            )
        if total < min_bytes:
            raise RestorerLoadError(f"Download from {url} is too small ({total} bytes)")
        if sha256 is not None and digest.hexdigest() != sha256.lower():
            raise RestorerLoadError(
                f"Checksum mismatch for {url}: expected {sha256}, got {digest.hexdigest()}"
            )
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        raise RestorerLoadError(f"Cannot download {url}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest
