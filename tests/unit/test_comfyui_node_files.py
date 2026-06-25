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


def test_artifact_removal_module_exports_one_node():
    from comfyui_nodes import artifact_removal as ar
    assert len(ar.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(ar.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/Artifact Removal"


def test_hdr_module_exports_one_node():
    from comfyui_nodes import hdr
    assert len(hdr.NODE_CLASS_MAPPINGS) == 1
    assert next(iter(hdr.NODE_CLASS_MAPPINGS.values())).CATEGORY == "RestoraX/HDR Conversion"


def test_stabilization_module_exports_two_nodes():
    from comfyui_nodes import stabilization as stab
    assert len(stab.NODE_CLASS_MAPPINGS) == 2
    for node_cls in stab.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Stabilization"


def test_audio_module_exports_three_nodes():
    from comfyui_nodes import audio
    assert len(audio.NODE_CLASS_MAPPINGS) == 3
    assert set(audio.NODE_CLASS_MAPPINGS) == set(audio.NODE_DISPLAY_NAME_MAPPINGS)
    for node_cls in audio.NODE_CLASS_MAPPINGS.values():
        assert node_cls.CATEGORY == "RestoraX/Audio"
        assert node_cls.RETURN_TYPES == ("AUDIO",)


def test_package_init_aggregates_all_25_nodes():
    import comfyui_nodes
    assert len(comfyui_nodes.NODE_CLASS_MAPPINGS) == 25
    assert set(comfyui_nodes.NODE_CLASS_MAPPINGS) == set(comfyui_nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert all(key.startswith("RestoraX_") for key in comfyui_nodes.NODE_CLASS_MAPPINGS)


def test_package_init_has_no_duplicate_node_keys_across_categories():
    import comfyui_nodes
    from comfyui_nodes import (
        artifact_removal, audio, colorization, deinterlacing, face_restoration,
        frame_interpolation, hdr, stabilization, super_resolution,
    )
    modules = [
        artifact_removal, audio, colorization, deinterlacing, face_restoration,
        frame_interpolation, hdr, stabilization, super_resolution,
    ]
    total_individual = sum(len(m.NODE_CLASS_MAPPINGS) for m in modules)
    assert total_individual == len(comfyui_nodes.NODE_CLASS_MAPPINGS) == 25
