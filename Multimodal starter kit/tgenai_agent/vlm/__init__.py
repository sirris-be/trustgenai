"""VLM subsystem package."""


# =================================== IMPORTS ==================================

# Standard Library
# None

# Third Party
# None

# Local
from tgenai_agent.vlm.annotation import annotate_points_on_image
from tgenai_agent.vlm.gemini_robotics_er import DETECT_COORDINATES_PROMPT, GeminiRoboticsER
from tgenai_agent.vlm.types import LabeledPoint2D, LabeledPoint3D, ObjectFindingResult, Vec3


# =================================== EXPORTS ===================================

__all__ = [
    "DETECT_COORDINATES_PROMPT",
    "GeminiRoboticsER",
    "LabeledPoint2D",
    "LabeledPoint3D",
    "ObjectFindingResult",
    "Vec3",
    "annotate_points_on_image",
]
