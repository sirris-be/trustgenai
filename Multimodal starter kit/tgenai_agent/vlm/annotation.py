"""Image annotation helpers for VLM detections."""


# =================================== IMPORTS ==================================

# Standard Library
# None

# Third Party
from PIL import Image, ImageDraw, ImageFont

# Local
from tgenai_agent.vlm.types import LabeledPoint2D


# =================================== CONSTANTS ==================================

POINT_FILL_COLOR = "#2862ff"
POINT_OUTLINE_COLOR = "#FFFFFF"
LABEL_FILL_COLOR = "#2862ff"
LABEL_TEXT_COLOR = "#FFFFFF"
POINT_RADIUS = 9
POINT_OUTLINE_WIDTH = 3
LABEL_PADDING_X = 12
LABEL_PADDING_Y = 7
LABEL_MARGIN_X = 14
LABEL_CORNER_RADIUS = 5
DEFAULT_FONT_SIZE = 16


# =============================== HELPER FUNCTIONS ===============================

def _load_annotation_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a readable label font, falling back to the default PIL font."""
    preferred_fonts = [
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in preferred_fonts:
        try:
            return ImageFont.truetype(font_path, size=DEFAULT_FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    """Measure a label's rendered width and height."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _compute_label_position(
    image_width: int,
    image_height: int,
    point_x: int,
    point_y: int,
    label_width: int,
    label_height: int,
) -> tuple[int, int]:
    """Place a label near a point while keeping it inside the image bounds."""
    label_x = point_x + POINT_RADIUS + LABEL_MARGIN_X
    label_y = point_y - (label_height // 2)

    max_label_x = max(image_width - label_width - 1, 0)
    max_label_y = max(image_height - label_height - 1, 0)

    if label_x > max_label_x:
        label_x = max(point_x - label_width - LABEL_MARGIN_X, 0)

    label_y = min(max(label_y, 0), max_label_y)
    return label_x, label_y


def _draw_point_marker(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """Draw a blue point marker with a white outline."""
    draw.ellipse(
        (x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS),
        fill=POINT_FILL_COLOR,
        outline=POINT_OUTLINE_COLOR,
        width=POINT_OUTLINE_WIDTH,
    )


def _draw_label_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    """Draw a rounded label badge similar to the Gemini demo visuals."""
    text_width, text_height = _measure_label(draw, text, font)
    badge_width = text_width + (LABEL_PADDING_X * 2)
    badge_height = text_height + (LABEL_PADDING_Y * 2)
    badge_box = (x, y, x + badge_width, y + badge_height)
    draw.rounded_rectangle(
        badge_box,
        radius=LABEL_CORNER_RADIUS,
        fill=LABEL_FILL_COLOR,
    )
    draw.text(
        (x + LABEL_PADDING_X, y + LABEL_PADDING_Y - 1),
        text,
        fill=LABEL_TEXT_COLOR,
        font=font,
    )


# ================================ MAIN FUNCTIONS ================================

def annotate_points_on_image(
    image: Image.Image,
    points: list[LabeledPoint2D],
) -> Image.Image:
    """Draw labeled normalized points onto a copy of the provided image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_annotation_font()

    width, height = annotated.size
    for point in points:
        y_norm, x_norm = point.point
        x = int(x_norm / 1000.0 * width)
        y = int(y_norm / 1000.0 * height)

        _draw_point_marker(draw, x, y)

        text_width, text_height = _measure_label(draw, point.label, font)
        badge_width = text_width + (LABEL_PADDING_X * 2)
        badge_height = text_height + (LABEL_PADDING_Y * 2)
        label_x, label_y = _compute_label_position(
            width,
            height,
            x,
            y,
            badge_width,
            badge_height,
        )
        _draw_label_badge(draw, label_x, label_y, point.label, font)

    return annotated
