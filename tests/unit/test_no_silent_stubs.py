"""
Canary: ensures no silent stub fallbacks exist in restorer source files.

Stubs previously hid load failures as silent passthrough (output identical
to input, job marked successful). Approved stubs are gone; any new _*Stub
class added under restorers/ will fail this test, forcing explicit review.
"""

from __future__ import annotations

import re
from pathlib import Path

_RESTORERS_DIR = Path(__file__).parent.parent.parent / "restorax" / "restorers"

_STUB_PATTERN = re.compile(r"^class (_\w*Stub)\b", re.MULTILINE)


def _find_stubs():
    """Return (relative_path, stub_name) for every stub class found."""
    results = []
    for py_file in _RESTORERS_DIR.rglob("*.py"):
        rel = py_file.relative_to(_RESTORERS_DIR).as_posix()
        text = py_file.read_text()
        for match in _STUB_PATTERN.finditer(text):
            results.append((rel, match.group(1)))
    return results


def test_no_silent_stubs():
    """No _*Stub passthrough classes allowed under restorers/."""
    found = _find_stubs()
    assert not found, (
        "Silent stub classes found in restorers/. Load failures must raise "
        "(RestorerLoadError/RuntimeError), not fall back to passthrough:\n"
        + "\n".join(f"  {rel}: {name}" for rel, name in found)
    )
