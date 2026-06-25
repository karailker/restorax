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


def test_colorization_module_exports_one_node():
    from comfyui_nodes import colorization as col
    assert len(col.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(col.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/Colorization"


def test_frame_interpolation_module_exports_one_node():
    from comfyui_nodes import frame_interpolation as fi
    assert len(fi.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(fi.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/Frame Interpolation"


def test_deinterlacing_module_exports_two_nodes():
    from comfyui_nodes import deinterlacing as dei
    assert len(dei.NODE_CLASS_MAPPINGS) == 2
    for node_cls in dei.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Deinterlacing"
