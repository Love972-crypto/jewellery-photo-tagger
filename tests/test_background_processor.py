import numpy as np
from PIL import Image, ImageDraw

import src.background_processor as background_module
from src.background_processor import (
    AI_FALLBACK_MODEL,
    BACKGROUND_OK,
    BACKGROUND_REVIEW_REQUIRED,
    BackgroundResult,
    MaskSafetyMetrics,
    choose_smart_hybrid_route,
    fuse_u2net_preservation_with_ai,
    needs_ai_refinement,
    remove_background,
)
from src.models import ProcessingSettings


def make_bgr_photo() -> np.ndarray:
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 35, 170, 135), fill="#b78935")
    return np.array(image)[:, :, ::-1].copy()


def test_settings_background_defaults():
    settings = ProcessingSettings()
    assert settings.remove_background is True
    assert settings.background_output_mode == "white_and_transparent"
    assert settings.ai_background_fallback_enabled is True
    assert settings.catalogue_layout_enabled is True


def test_residue_or_missing_component_recommends_ai_refinement():
    clean = MaskSafetyMetrics(0.2, 0.2, 0.0, 0.0, 0.0, 2, 2, 0, 0, 0.0001, 0.00005)
    residue = MaskSafetyMetrics(0.2, 0.2, 0.0, 0.0, 0.0, 2, 2, 0, 0, 0.001, 0.00005)
    missing = MaskSafetyMetrics(0.2, 0.2, 0.0, 0.0, 0.0, 2, 2, 0, 1, 0.0, 0.0)

    assert not needs_ai_refinement(BackgroundResult(BACKGROUND_OK, safety_metrics=clean))
    assert needs_ai_refinement(BackgroundResult(BACKGROUND_OK, safety_metrics=residue))
    assert needs_ai_refinement(BackgroundResult(BACKGROUND_OK, safety_metrics=missing))
    assert needs_ai_refinement(BackgroundResult(BACKGROUND_REVIEW_REQUIRED))


def test_smart_hybrid_uses_birefnet_only_for_safe_uniform_background():
    source = Image.new("RGB", (360, 280), "white")
    ImageDraw.Draw(source).ellipse((90, 55, 270, 230), fill="#b78935")
    source_rgb = np.array(source)
    source_bgr = source_rgb[:, :, ::-1].copy()
    alpha = np.zeros((280, 360), dtype=np.uint8)
    alpha[55:231, 90:271] = 255
    metrics = MaskSafetyMetrics(0.31, 0.31, 0.0, 0.0, 0.0, 1, 1, 0, 0, 0.0001, 0.00005)
    result = BackgroundResult(
        BACKGROUND_OK,
        source_rgba=np.dstack((source_rgb, alpha)),
        safety_metrics=metrics,
    )

    decision = choose_smart_hybrid_route(source_bgr, result)

    assert decision.route == "birefnet_only"
    assert not decision.use_u2net_preservation
    assert decision.gradient_p90 < 38.0


def test_smart_hybrid_adds_u2net_for_textured_background():
    source = Image.new("RGB", (360, 280), "#b9afa4")
    draw = ImageDraw.Draw(source)
    for x in range(0, 360, 9):
        draw.line((x, 0, x, 279), fill="#6e6258", width=3)
    draw.ellipse((90, 55, 270, 230), fill="#b78935")
    source_rgb = np.array(source)
    source_bgr = source_rgb[:, :, ::-1].copy()
    alpha = np.zeros((280, 360), dtype=np.uint8)
    alpha[55:231, 90:271] = 255
    metrics = MaskSafetyMetrics(0.31, 0.31, 0.0, 0.0, 0.0, 1, 1, 0, 0, 0.0001, 0.00005)
    result = BackgroundResult(
        BACKGROUND_OK,
        source_rgba=np.dstack((source_rgb, alpha)),
        safety_metrics=metrics,
    )

    decision = choose_smart_hybrid_route(source_bgr, result)

    assert decision.route == "birefnet_plus_u2net"
    assert decision.use_u2net_preservation
    assert any("textured" in reason for reason in decision.reasons)


def _mock_u2net_with_gold_object(input_image, alpha_matting, model_name):
    output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
    ImageDraw.Draw(output).ellipse((45, 55, 175, 205), fill=(183, 137, 53, 255))
    return output


def test_broad_tag_preservation_with_wood_residue_recommends_ai_refinement(monkeypatch):
    source = Image.new("RGB", (360, 260), "#81786d")
    draw = ImageDraw.Draw(source)
    draw.ellipse((45, 55, 175, 205), fill="#b78935")
    draw.rectangle((195, 108, 302, 182), fill="white")
    draw.text((220, 132), "122529", fill="black")
    source_bgr = np.array(source)[:, :, ::-1].copy()
    preserve_mask = np.zeros(source_bgr.shape[:2], dtype=np.uint8)
    preserve_mask[78:222, 165:332] = 255
    monkeypatch.setattr(background_module, "_remove_with_rembg", _mock_u2net_with_gold_object)

    result = remove_background(
        source_bgr,
        preserve_mask=preserve_mask,
        catalogue_layout=False,
    )

    assert result.status == BACKGROUND_OK
    assert needs_ai_refinement(result)
    assert result.ai_refinement_reasons
    assert "tag preservation includes likely background" in result.notes
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[90, 180, 3]) > 0


def test_tight_clean_tag_preservation_does_not_recommend_ai_refinement(monkeypatch):
    source = Image.new("RGB", (360, 260), "#81786d")
    draw = ImageDraw.Draw(source)
    draw.ellipse((45, 55, 175, 205), fill="#b78935")
    draw.rectangle((195, 108, 302, 182), fill="white")
    draw.text((220, 132), "122529", fill="black")
    source_bgr = np.array(source)[:, :, ::-1].copy()
    preserve_mask = np.zeros(source_bgr.shape[:2], dtype=np.uint8)
    preserve_mask[108:183, 195:303] = 255
    monkeypatch.setattr(background_module, "_remove_with_rembg", _mock_u2net_with_gold_object)

    result = remove_background(
        source_bgr,
        preserve_mask=preserve_mask,
        catalogue_layout=False,
    )

    assert result.status == BACKGROUND_OK
    assert not result.ai_refinement_reasons
    assert not needs_ai_refinement(result)


def test_gold_and_red_jewellery_touching_tag_do_not_look_like_floor_residue(monkeypatch):
    source = Image.new("RGB", (360, 260), "white")
    draw = ImageDraw.Draw(source)
    draw.ellipse((45, 55, 205, 215), fill="#b78935")
    draw.rectangle((190, 108, 302, 182), fill="white")
    draw.line((170, 130, 225, 130), fill="#991b35", width=10)
    draw.text((220, 140), "122529", fill="black")
    source_bgr = np.array(source)[:, :, ::-1].copy()
    preserve_mask = np.zeros(source_bgr.shape[:2], dtype=np.uint8)
    preserve_mask[103:188, 185:308] = 255
    monkeypatch.setattr(background_module, "_remove_with_rembg", _mock_u2net_with_gold_object)

    result = remove_background(
        source_bgr,
        preserve_mask=preserve_mask,
        catalogue_layout=False,
    )

    assert result.status == BACKGROUND_OK
    assert not result.ai_refinement_reasons
    assert not needs_ai_refinement(result)


def test_ai_fallback_uses_conservative_raw_matte_cleanup(monkeypatch):
    captured = {}

    def fake_remove(input_image, alpha_matting, model_name):
        output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
        ImageDraw.Draw(output).ellipse((55, 25, 185, 155), fill=(183, 137, 53, 255))
        return output

    real_cleanup = background_module._clean_alpha_matte

    def capture_cleanup(image_rgba, source_bgr=None):
        captured["source_bgr"] = source_bgr
        return real_cleanup(image_rgba, source_bgr=source_bgr)

    monkeypatch.setattr(background_module, "_remove_with_rembg", fake_remove)
    monkeypatch.setattr(background_module, "_clean_alpha_matte", capture_cleanup)

    result = remove_background(make_bgr_photo(), model_name=AI_FALLBACK_MODEL, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert captured["source_bgr"] is None


def test_background_processor_returns_white_and_transparent_from_mock_remover():
    def remover(input_image):
        output = input_image.convert("RGBA")
        alpha = Image.new("L", input_image.size, 0)
        draw = ImageDraw.Draw(alpha)
        draw.ellipse((55, 25, 185, 155), fill=255)
        output.putalpha(alpha)
        return output

    result = remove_background(make_bgr_photo(), remover=remover)

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert result.transparent_rgba.shape[:2] == (1500, 1200)
    assert result.transparent_rgba.shape[2] == 4
    assert int(result.transparent_rgba[0, 0, 3]) == 0
    assert result.white_bgr is not None
    assert result.white_bgr.shape[:2] == (1500, 1200)
    assert result.white_bgr.shape[2] == 3


def test_background_processor_rejects_empty_alpha_mask():
    def remover(input_image):
        output = input_image.convert("RGBA")
        output.putalpha(Image.new("L", input_image.size, 0))
        return output

    result = remove_background(make_bgr_photo(), remover=remover)

    assert result.status == BACKGROUND_REVIEW_REQUIRED
    assert "too small" in result.notes


def test_background_processor_removes_grey_residue_far_from_product():
    def remover(input_image):
        output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(output)
        draw.ellipse((70, 35, 170, 135), fill=(183, 137, 53, 255))
        draw.rectangle((188, 58, 235, 145), fill=(90, 90, 90, 200))
        return output

    result = remove_background(make_bgr_photo(), remover=remover, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[90, 210, 3]) == 0
    assert result.white_bgr is not None
    assert result.white_bgr[90, 210].tolist() == [255, 255, 255]
    assert "Removed background residue" in result.notes


def test_background_processor_restores_colored_object_hole_from_source_photo():
    image = Image.new("RGB", (260, 180), "#a79f92")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 45, 145, 135), fill="#b78935")
    source_bgr = np.array(image)[:, :, ::-1].copy()

    def remover(input_image):
        output = input_image.convert("RGBA")
        alpha = Image.new("L", input_image.size, 0)
        draw_alpha = ImageDraw.Draw(alpha)
        draw_alpha.ellipse((45, 45, 145, 135), fill=255)
        draw_alpha.ellipse((78, 72, 112, 106), fill=0)
        draw_alpha.rectangle((165, 55, 250, 145), fill=220)
        output.putalpha(alpha)
        return output

    result = remove_background(source_bgr, remover=remover, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[90, 95, 3]) >= 220
    assert result.transparent_rgba[90, 95, :3].tolist() == source_bgr[90, 95, ::-1].tolist()
    assert int(result.transparent_rgba[90, 205, 3]) == 0


def test_background_processor_routes_catastrophic_jewellery_loss_to_review(monkeypatch):
    image = Image.new("RGB", (520, 300), "#9b9387")
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 70, 150, 190), fill="#b78935")
    bead_positions = [(180 + (index % 10) * 30, 35 + (index // 10) * 55) for index in range(30)]
    for x, y in bead_positions:
        draw.ellipse((x, y, x + 20, y + 20), fill="#c79b3f")
    source_bgr = np.array(image)[:, :, ::-1].copy()

    def fake_remove(input_image, alpha_matting, model_name):
        output = input_image.convert("RGBA")
        alpha = Image.new("L", input_image.size, 0)
        alpha_draw = ImageDraw.Draw(alpha)
        alpha_draw.ellipse((30, 70, 150, 190), fill=255)
        for x, y in bead_positions:
            alpha_draw.ellipse((x, y, x + 20, y + 20), fill=95)
        output.putalpha(alpha)
        return output

    monkeypatch.setattr(background_module, "_remove_with_rembg", fake_remove)
    result = remove_background(source_bgr, catalogue_layout=False)

    assert result.status == BACKGROUND_REVIEW_REQUIRED
    assert result.safety_metrics is not None
    assert not result.safety_metrics.safe
    assert result.safety_metrics.removed_source_supported_ratio > 0.008
    assert "Jewellery preservation risk" in result.notes


def test_background_processor_routes_multiple_coloured_components_missing_from_raw_mask(monkeypatch):
    image = Image.new("RGB", (320, 210), "#a8a097")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 45, 155, 155), fill="#b78935")
    for y in (35, 85, 135):
        draw.ellipse((235, y, 275, y + 40), fill="#9d1434")
    source_bgr = np.array(image)[:, :, ::-1].copy()

    def fake_remove(input_image, alpha_matting, model_name):
        output = input_image.convert("RGBA")
        alpha = Image.new("L", input_image.size, 0)
        ImageDraw.Draw(alpha).ellipse((45, 45, 155, 155), fill=255)
        output.putalpha(alpha)
        return output

    monkeypatch.setattr(background_module, "_remove_with_rembg", fake_remove)
    result = remove_background(source_bgr, catalogue_layout=False)

    assert result.status == BACKGROUND_REVIEW_REQUIRED
    assert result.safety_metrics is not None
    assert result.safety_metrics.missing_source_components >= 3
    assert "missing from the mask" in result.notes


def test_background_processor_residue_does_not_force_review():
    image = Image.new("RGB", (320, 210), "#9b9387")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 45, 155, 155), fill="#b78935")
    source_bgr = np.array(image)[:, :, ::-1].copy()

    def remover(input_image):
        output = input_image.convert("RGBA")
        alpha = Image.new("L", input_image.size, 0)
        alpha_draw = ImageDraw.Draw(alpha)
        alpha_draw.ellipse((45, 45, 155, 155), fill=255)
        alpha_draw.rectangle((152, 72, 215, 130), fill=255)
        output.putalpha(alpha)
        return output

    result = remove_background(source_bgr, remover=remover, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert result.safety_metrics is not None
    assert result.safety_metrics.largest_residue_component_ratio > 0.00025


def test_background_processor_removes_isolated_low_confidence_floor_line():
    source = Image.new("RGB", (360, 300), "white")
    ImageDraw.Draw(source).ellipse((75, 65, 245, 235), fill="#b78935")
    source_bgr = np.array(source)[:, :, ::-1].copy()

    def remover(input_image):
        output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(output)
        draw.ellipse((75, 65, 245, 235), fill=(183, 137, 53, 255))
        draw.line((290, 172, 302, 299), fill=(80, 66, 51, 10), width=7)
        draw.line((292, 172, 300, 299), fill=(80, 66, 51, 120), width=3)
        for line_y in range(172, 300, 12):
            line_x = 292 + round((line_y - 172) * 8 / 127)
            draw.point((line_x, line_y), fill=(80, 66, 51, 240))
        return output

    result = remove_background(source_bgr, remover=remover, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[245, 297, 3]) == 0
    assert int(result.transparent_rgba[245, 300, 3]) == 0
    assert int(result.transparent_rgba[150, 155, 3]) == 255
    assert "Removed isolated background line" in result.notes


def test_background_processor_keeps_solid_isolated_jewellery_chain():
    source = Image.new("RGB", (360, 300), "white")
    ImageDraw.Draw(source).ellipse((75, 65, 245, 235), fill="#b78935")
    source_bgr = np.array(source)[:, :, ::-1].copy()

    def remover(input_image):
        output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(output)
        draw.ellipse((75, 65, 245, 235), fill=(183, 137, 53, 255))
        draw.line((315, 120, 315, 299), fill=(196, 146, 48, 255), width=5)
        return output

    result = remove_background(source_bgr, remover=remover, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[245, 315, 3]) == 255


def test_connected_weak_floor_seam_tail_is_removed_without_touching_product():
    alpha_image = Image.new("L", (360, 400), 0)
    draw = ImageDraw.Draw(alpha_image)
    draw.ellipse((45, 80, 150, 280), fill=255)
    draw.ellipse((210, 80, 315, 280), fill=255)
    draw.line((177, 180, 183, 399), fill=120, width=3)
    for line_y in range(190, 400, 14):
        line_x = 177 + round((line_y - 180) * 6 / 219)
        draw.point((line_x, line_y), fill=240)
    alpha = np.array(alpha_image)

    cleaned, removed = background_module._remove_weak_linear_edge_tails(alpha)

    assert removed == 1
    assert int(cleaned[350, 182]) == 0
    assert int(cleaned[220, 179]) == 0
    assert int(cleaned[150, 100]) == 255


def test_connected_solid_chain_tail_is_preserved():
    alpha_image = Image.new("L", (360, 400), 0)
    draw = ImageDraw.Draw(alpha_image)
    draw.line((160, 210, 166, 399), fill=255, width=5)
    draw.ellipse((75, 80, 245, 230), fill=255)
    alpha = np.array(alpha_image)

    cleaned, removed = background_module._remove_weak_linear_edge_tails(alpha)

    assert removed == 0
    assert int(cleaned[350, 164]) == 255


def test_default_u2net_uses_proven_pre_windows_model(monkeypatch):
    captured = {}

    def fake_remove(input_image, alpha_matting, model_name):
        captured["model_name"] = model_name
        output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(output)
        draw.ellipse((55, 30, 165, 150), fill=(183, 137, 53, 255))
        draw.rectangle((175, 45, 230, 105), fill=(248, 246, 238, 180))
        draw.text((185, 62), "122358", fill=(20, 20, 20, 220))
        return output

    monkeypatch.setattr(background_module, "_remove_with_rembg", fake_remove)

    result = remove_background(make_bgr_photo(), catalogue_layout=False)

    assert captured["model_name"] == "u2net"
    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[90, 110, 3]) == 255


def test_explicit_tag_mask_restores_source_pixels_and_alpha():
    source = make_bgr_photo()
    preserve_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    preserve_mask[40:100, 165:230] = 255

    def remover(input_image):
        output = Image.new("RGBA", input_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(output)
        draw.ellipse((55, 25, 185, 155), fill=(183, 137, 53, 255))
        return output

    result = remove_background(
        source,
        remover=remover,
        preserve_mask=preserve_mask,
        catalogue_layout=False,
    )

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[60, 200, 3]) == 255
    assert result.transparent_rgba[60, 200, :3].tolist() == source[60, 200, ::-1].tolist()


def _background_result_from_alpha(source_rgb: np.ndarray, alpha: np.ndarray) -> BackgroundResult:
    rgba = np.dstack((source_rgb, alpha.astype(np.uint8)))
    return BackgroundResult(BACKGROUND_OK, source_rgba=rgba)


def test_hybrid_keeps_birefnet_base_and_rejects_u2net_wood_halo():
    source = Image.new("RGB", (320, 240), "#81786d")
    ImageDraw.Draw(source).ellipse((90, 55, 225, 190), fill="#c28b25")
    source_rgb = np.array(source)
    source_bgr = source_rgb[:, :, ::-1].copy()

    ai_alpha_image = Image.new("L", source.size, 0)
    ImageDraw.Draw(ai_alpha_image).ellipse((90, 55, 225, 190), fill=255)
    ai_alpha = np.array(ai_alpha_image)
    u2net_alpha_image = Image.new("L", source.size, 0)
    ImageDraw.Draw(u2net_alpha_image).ellipse((65, 30, 250, 215), fill=255)
    u2net_alpha = np.array(u2net_alpha_image)

    result = fuse_u2net_preservation_with_ai(
        source_bgr,
        _background_result_from_alpha(source_rgb, u2net_alpha),
        _background_result_from_alpha(source_rgb, ai_alpha),
        catalogue_layout=False,
    )

    assert result.status == BACKGROUND_OK
    assert result.source_rgba is not None
    assert int(result.source_rgba[120, 75, 3]) == 0
    assert int(result.source_rgba[120, 150, 3]) == 255


def test_hybrid_does_not_replace_existing_birefnet_opacity_with_u2net():
    source = Image.new("RGB", (260, 200), "white")
    ImageDraw.Draw(source).ellipse((70, 40, 190, 160), fill="#c28b25")
    source_rgb = np.array(source)
    source_bgr = source_rgb[:, :, ::-1].copy()
    ai_alpha = np.zeros((200, 260), dtype=np.uint8)
    ai_alpha[40:161, 70:191] = 120
    u2net_alpha = np.zeros_like(ai_alpha)
    u2net_alpha[35:166, 65:196] = 255

    result = fuse_u2net_preservation_with_ai(
        source_bgr,
        _background_result_from_alpha(source_rgb, u2net_alpha),
        _background_result_from_alpha(source_rgb, ai_alpha),
        catalogue_layout=False,
    )

    assert result.source_rgba is not None
    assert int(result.source_rgba[100, 130, 3]) == 120


def test_hybrid_restores_only_connected_saturated_jewellery_edge():
    source = Image.new("RGB", (280, 210), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((70, 55, 175, 155), fill="#c28b25")
    draw.rectangle((176, 95, 178, 115), fill="#087f55")
    draw.ellipse((230, 80, 250, 100), fill="#087f55")
    source_rgb = np.array(source)
    source_bgr = source_rgb[:, :, ::-1].copy()
    ai_alpha = np.zeros((210, 280), dtype=np.uint8)
    ai_alpha[55:156, 70:176] = 255
    u2net_alpha = ai_alpha.copy()
    u2net_alpha[95:116, 176:179] = 255
    u2net_alpha[80:101, 230:251] = 255

    result = fuse_u2net_preservation_with_ai(
        source_bgr,
        _background_result_from_alpha(source_rgb, u2net_alpha),
        _background_result_from_alpha(source_rgb, ai_alpha),
        catalogue_layout=False,
    )

    assert result.source_rgba is not None
    assert int(result.source_rgba[105, 177, 3]) == 255
    assert int(result.source_rgba[90, 240, 3]) == 0


def test_hybrid_removes_neutral_floor_patch_outside_physical_tag():
    source = Image.new("RGB", (320, 240), "#8a8177")
    draw = ImageDraw.Draw(source)
    draw.ellipse((55, 70, 155, 170), fill="#c28b25")
    draw.rectangle((215, 65, 294, 115), fill="white")
    source_rgb = np.array(source)
    source_bgr = source_rgb[:, :, ::-1].copy()

    ai_alpha = np.zeros((240, 320), dtype=np.uint8)
    ai_alpha[70:171, 55:156] = 255
    ai_alpha[60:121, 210:300] = 255
    u2net_alpha = ai_alpha.copy()
    tag_mask = np.zeros_like(ai_alpha)
    tag_mask[65:116, 215:295] = 255

    result = fuse_u2net_preservation_with_ai(
        source_bgr,
        _background_result_from_alpha(source_rgb, u2net_alpha),
        _background_result_from_alpha(source_rgb, ai_alpha),
        catalogue_layout=False,
        tag_preserve_mask=tag_mask,
    )

    assert result.source_rgba is not None
    assert int(result.source_rgba[90, 250, 3]) == 255
    assert int(result.source_rgba[62, 250, 3]) == 0
    assert int(result.source_rgba[120, 105, 3]) == 255
    assert "immediately outside the physical tag" in result.notes
