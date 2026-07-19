"""Image style preset settings and prompt helpers."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.persistence.repositories import PersistenceRepositories

IMAGE_STYLE_PRESET_SETTING = "image_style_preset"
DEFAULT_IMAGE_STYLE_PRESET = "none"
_SAVE_IMAGE_STYLE_PRESET_PREFIX = f"{IMAGE_STYLE_PRESET_SETTING}:save:"


@dataclass(frozen=True)
class ImageStylePreset:
    id: str
    label: str
    instruction: str


_IMAGE_STYLE_PRESETS = (
    ImageStylePreset(id="none", label="No preset", instruction=""),
    ImageStylePreset(
        id="realistic",
        label="Realistic",
        instruction=(
            "Use a photoreal image style with natural lighting, real-world "
            "materials, grounded anatomy, and cinematic composition."
        ),
    ),
    ImageStylePreset(
        id="anime",
        label="Anime",
        instruction=(
            "Use a polished anime illustration style with expressive faces, "
            "clean linework, saturated colors, and dynamic composition."
        ),
    ),
    ImageStylePreset(
        id="cartoon",
        label="Cartoon",
        instruction=(
            "Use a stylized cartoon illustration style with bold shapes, clear "
            "silhouettes, lively color, and readable expressions."
        ),
    ),
    ImageStylePreset(
        id="cinematic",
        label="Cinematic",
        instruction=(
            "Use a cinematic visual style with dramatic lighting, film still "
            "composition, atmospheric depth, and filmic color grading."
        ),
    ),
    ImageStylePreset(
        id="concept_art",
        label="Concept Art",
        instruction=(
            "Use a concept art style with clear design shapes, strong "
            "silhouettes, polished worldbuilding details, and production-ready mood."
        ),
    ),
    ImageStylePreset(
        id="digital_painting",
        label="Digital Painting",
        instruction=(
            "Use a digital painting style with painterly brushwork, rich color, "
            "controlled edges, and refined lighting."
        ),
    ),
    ImageStylePreset(
        id="watercolor",
        label="Watercolor",
        instruction=(
            "Use a watercolor style with translucent washes, soft edges, paper "
            "texture, and gentle color blending."
        ),
    ),
    ImageStylePreset(
        id="oil_painting",
        label="Oil Painting",
        instruction=(
            "Use an oil painting style with layered brushstrokes, textured "
            "paint, deep color, and classical lighting."
        ),
    ),
    ImageStylePreset(
        id="comic_book",
        label="Comic Book",
        instruction=(
            "Use a comic book illustration style with bold inks, dramatic "
            "panel-like composition, crisp shadows, and vivid flat color."
        ),
    ),
    ImageStylePreset(
        id="colored_pencil",
        label="Colored Pencil",
        instruction=(
            "Use a colored pencil style with visible pencil texture, layered "
            "strokes, soft shading, and hand-drawn detail."
        ),
    ),
    ImageStylePreset(
        id="sketch",
        label="Sketch",
        instruction=(
            "Use a sketch style with confident graphite lines, loose "
            "construction marks, minimal color, and expressive shading."
        ),
    ),
    ImageStylePreset(
        id="ink",
        label="Ink",
        instruction=(
            "Use an ink illustration style with precise linework, strong "
            "contrast, controlled hatching, and clean negative space."
        ),
    ),
    ImageStylePreset(
        id="pixel_art",
        label="Pixel Art",
        instruction=(
            "Use a pixel art style with crisp low-resolution forms, limited "
            "palette, readable silhouettes, and no smoothing."
        ),
    ),
    ImageStylePreset(
        id="three_d_render",
        label="3D Render",
        instruction=(
            "Use a 3D render style with modeled forms, realistic materials, "
            "volumetric lighting, and depth of field."
        ),
    ),
    ImageStylePreset(
        id="low_poly",
        label="Low Poly",
        instruction=(
            "Use a low poly 3D style with faceted geometry, simplified forms, "
            "clean color planes, and stylized lighting."
        ),
    ),
)
_IMAGE_STYLE_PRESETS_BY_ID = {preset.id: preset for preset in _IMAGE_STYLE_PRESETS}


def image_style_presets() -> tuple[ImageStylePreset, ...]:
    return _IMAGE_STYLE_PRESETS


def image_style_preset_options() -> tuple[str, ...]:
    return tuple(preset.id for preset in _IMAGE_STYLE_PRESETS)


def sanitize_image_style_preset(value: object) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower().replace("-", "_")
        if candidate in _IMAGE_STYLE_PRESETS_BY_ID:
            return candidate
    return DEFAULT_IMAGE_STYLE_PRESET


def save_image_style_preset_setting_key(save_id: str) -> str:
    return f"{_SAVE_IMAGE_STYLE_PRESET_PREFIX}{save_id}"


def selected_image_style_preset(
    repositories: PersistenceRepositories,
    *,
    save_id: str | None = None,
) -> str:
    if save_id is None:
        return DEFAULT_IMAGE_STYLE_PRESET
    return sanitize_image_style_preset(
        repositories.get_scoped_setting(
            scope="save",
            scope_id=save_id,
            key=IMAGE_STYLE_PRESET_SETTING,
        )
    )


def image_style_preset_label(preset_id: object) -> str:
    preset = _IMAGE_STYLE_PRESETS_BY_ID[
        sanitize_image_style_preset(preset_id)
    ]
    return preset.label


def apply_image_style_preset(prompt: str, *, preset_id: object) -> str:
    base_prompt = prompt.strip()
    if not base_prompt:
        return ""
    preset = _IMAGE_STYLE_PRESETS_BY_ID[
        sanitize_image_style_preset(preset_id)
    ]
    if not preset.instruction:
        return base_prompt
    return f"{base_prompt}\n\nStyle preset: {preset.label}. {preset.instruction}"
