"""
Canary: ensures no silent stub fallbacks exist outside approved exceptions.

Scans restorer source files for _*Stub classes. Only the approved audio
passthrough stubs (demucs, rnnoise, voicefixer) are allowed. Any new stub
added outside audio/ will fail this test, forcing an explicit review.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_RESTORERS_DIR = Path(__file__).parent.parent.parent / "restorax" / "restorers"

# Approved silent stubs — audio restorers keep passthrough stubs
# because audio restoration is optional and graceful degradation is intentional.
_APPROVED_STUBS = {
    "audio/demucs.py": ["_DemucsStub"],
    "audio/rnnoise.py": ["_RNNoiseStub"],
    "audio/voicefixer.py": ["_VoiceFixerStub"],
}

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


def test_no_unapproved_silent_stubs():
    """All _*Stub classes must be in the approved list."""
    unapproved = []
    for rel_path, stub_name in _find_stubs():
        approved_names = _APPROVED_STUBS.get(rel_path, [])
        if stub_name not in approved_names:
            unapproved.append(f"{rel_path}: {stub_name}")

    assert not unapproved, (
        "Unapproved silent stub classes found. Either remove the stub and raise "
        "RestorerLoadError, or add it to _APPROVED_STUBS with justification:\n"
        + "\n".join(f"  {u}" for u in unapproved)
    )


def test_approved_stubs_still_exist():
    """Verify approved stubs haven't been accidentally removed."""
    found = {rel: name for rel, name in _find_stubs()}
    for rel_path, stub_names in _APPROVED_STUBS.items():
        for stub_name in stub_names:
            # Only check if the file exists
            if (_RESTORERS_DIR / rel_path).exists():
                assert found.get(rel_path) == stub_name or stub_name in str(found), (
                    f"Expected approved stub {stub_name} in {rel_path} — "
                    "update _APPROVED_STUBS if it was intentionally removed."
                )
