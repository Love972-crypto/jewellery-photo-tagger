import numpy as np
from PIL import Image, ImageDraw

from src.background_processor import BACKGROUND_OK, BACKGROUND_REVIEW_REQUIRED, remove_background
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
    assert settings.catalogue_layout_enabled is True


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
        draw.ellipse((48, 44, 122, 130), fill=(183, 137, 53, 255))
        draw.rectangle((150, 58, 230, 145), fill=(90, 90, 90, 200))
        return output

    result = remove_background(make_bgr_photo(), remover=remover, catalogue_layout=False)

    assert result.status == BACKGROUND_OK
    assert result.transparent_rgba is not None
    assert int(result.transparent_rgba[90, 190, 3]) == 0
    assert result.white_bgr is not None
    assert result.white_bgr[90, 190].tolist() == [255, 255, 255]
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
    assert int(result.transparent_rgba[90, 205, 3]) == 0
