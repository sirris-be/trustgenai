"""Shared data models for the VLM subsystem."""


# =================================== IMPORTS ==================================

# Standard Library
# None

# Third Party
from pydantic import BaseModel

# Local
# None


# ==================================== TYPES ====================================

class Vec3(BaseModel):
    """A 3D point in camera/world coordinates."""

    x: float
    y: float
    z: float


class LabeledPoint2D(BaseModel):
    """A detected object location in the image."""

    point: list[float]  # [y, x] normalized to 0-1000
    label: str


class LabeledPoint3D(BaseModel):
    """A detected object location with 3D coordinates."""

    point: Vec3
    label: str


class ObjectFindingResult(BaseModel):
    """Result of an object-finding VLM call."""

    objects: list[LabeledPoint2D]
    description: str
