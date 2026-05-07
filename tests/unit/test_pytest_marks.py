def test_requires_weights_mark_registered(pytestconfig):
    raw = pytestconfig.getini("markers")
    # Each entry is either a string "name: description" or a MarkerInfo object
    marks = set()
    for m in raw:
        name = m.name if hasattr(m, "name") else m.split(":")[0].strip()
        marks.add(name)
    assert "requires_weights" in marks
    assert "requires_assets" in marks
    assert "benchmark" in marks
