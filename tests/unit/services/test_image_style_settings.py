from bragi.services.image_style_settings import (
    DEFAULT_IMAGE_STYLE_PRESET,
    IMAGE_STYLE_PRESET_SETTING,
    apply_image_style_preset,
    image_style_preset_label,
    image_style_preset_options,
    image_style_presets,
    sanitize_image_style_preset,
)

EXPECTED_IMAGE_STYLE_PRESET_IDS = (
    "none",
    "realistic",
    "anime",
    "cartoon",
    "cinematic",
    "concept_art",
    "digital_painting",
    "watercolor",
    "oil_painting",
    "comic_book",
    "colored_pencil",
    "sketch",
    "ink",
    "pixel_art",
    "three_d_render",
    "low_poly",
)


def test_image_style_preset_options_include_default_and_supported_styles() -> None:
    assert IMAGE_STYLE_PRESET_SETTING == "image_style_preset"
    assert DEFAULT_IMAGE_STYLE_PRESET == "realistic"
    assert image_style_preset_options() == EXPECTED_IMAGE_STYLE_PRESET_IDS
    preset_ids = tuple(preset.id for preset in image_style_presets())
    assert preset_ids == EXPECTED_IMAGE_STYLE_PRESET_IDS


def test_image_style_preset_labels_are_stable_and_provider_agnostic() -> None:
    assert image_style_preset_label("concept_art") == "Concept Art"
    assert image_style_preset_label("oil_painting") == "Oil Painting"
    assert image_style_preset_label("three_d_render") == "3D Render"
    assert image_style_preset_label("unknown") == "Realistic"


def test_sanitize_image_style_preset_rejects_unknown_values() -> None:
    assert sanitize_image_style_preset("anime") == "anime"
    assert sanitize_image_style_preset("  cartoon  ") == "cartoon"
    assert sanitize_image_style_preset("colored_pencil") == "colored_pencil"
    assert sanitize_image_style_preset("pixel-art") == "pixel_art"
    assert sanitize_image_style_preset("oil painting") == "realistic"
    assert sanitize_image_style_preset(True) == "realistic"


def test_apply_image_style_preset_appends_selected_style_instruction() -> None:
    styled = apply_image_style_preset(
        "moonlit keep over a frozen river",
        preset_id="realistic",
    )

    assert styled.startswith("moonlit keep over a frozen river\n\n")
    assert "Style preset: Realistic." in styled
    assert "photoreal" in styled


def test_apply_image_style_preset_appends_new_style_instruction() -> None:
    styled = apply_image_style_preset(
        "moonlit keep over a frozen river",
        preset_id="pixel-art",
    )

    assert styled.startswith("moonlit keep over a frozen river\n\n")
    assert "Style preset: Pixel Art." in styled
    assert "limited palette" in styled
    assert "no smoothing" in styled


def test_apply_image_style_preset_leaves_none_preset_neutral() -> None:
    prompt = "moonlit keep over a frozen river"

    assert apply_image_style_preset(prompt, preset_id="none") == prompt
    assert apply_image_style_preset("   ", preset_id="anime") == ""
