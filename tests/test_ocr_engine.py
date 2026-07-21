import numpy as np
from types import SimpleNamespace

from src.ocr_engine import EasyOCREngine, _ensure_torchvision_compatibility


class CapturingReader:
    def __init__(self):
        self.options = None

    def readtext(self, image, **options):
        self.options = options
        return []


def test_full_frame_ocr_uses_high_resolution_detection_profile():
    reader = CapturingReader()
    engine = EasyOCREngine()
    engine._reader = reader

    engine.read_text(np.zeros((1500, 1080, 3), dtype=np.uint8), source_crop="fallback_full")

    assert reader.options["canvas_size"] == 1800
    assert reader.options["mag_ratio"] == 1.2
    assert reader.options["min_size"] == 8
    assert reader.options["text_threshold"] == 0.45


def test_detected_crop_uses_high_accuracy_numeric_profile():
    reader = CapturingReader()
    engine = EasyOCREngine()
    engine._reader = reader

    engine.read_text(np.zeros((220, 420, 3), dtype=np.uint8), source_crop="tag_crop_1")

    assert reader.options["canvas_size"] == 1280
    assert reader.options["mag_ratio"] == 1.35
    assert reader.options["text_threshold"] == 0.45
    assert reader.options["contrast_ths"] == 0.05


def test_uncertain_crop_verification_uses_independent_beam_decoder():
    reader = CapturingReader()
    engine = EasyOCREngine()
    engine._reader = reader

    engine.read_text_verification(np.zeros((220, 420, 3), dtype=np.uint8))

    assert reader.options["decoder"] == "beamsearch"
    assert reader.options["beamWidth"] == 5


def test_torchvision_compatibility_skips_only_missing_optional_fake_ops_and_restores():
    registrations = []

    def original_register_fake(operator_name, *args, **kwargs):
        registrations.append(operator_name)
        return lambda function: function

    def dispatch_check(operator_name, dispatch_key):
        del dispatch_key
        if operator_name in {"torchvision::nms", "torchvision::qnms"}:
            raise RuntimeError("operator does not exist")
        return False

    fake_torch = SimpleNamespace(
        library=SimpleNamespace(register_fake=original_register_fake),
        _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=dispatch_check),
    )

    restore = _ensure_torchvision_compatibility(fake_torch)
    decorator = fake_torch.library.register_fake("torchvision::qnms")
    marker = object()

    assert decorator(marker) is marker
    assert registrations == []

    restore()
    fake_torch.library.register_fake("other::op")
    assert registrations == ["other::op"]
