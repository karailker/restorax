def test_super_resolution_module_exports_ten_nodes():
    from comfyui_nodes import super_resolution as sr
    assert len(sr.NODE_CLASS_MAPPINGS) == 10
    assert set(sr.NODE_CLASS_MAPPINGS) == set(sr.NODE_DISPLAY_NAME_MAPPINGS)
    for node_cls in sr.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Super Resolution"


def test_face_restoration_module_exports_four_nodes():
    from comfyui_nodes import face_restoration as fr
    assert len(fr.NODE_CLASS_MAPPINGS) == 4
    for node_cls in fr.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Face Restoration"
