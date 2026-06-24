"""Native Gemini Robotics-ER wrapper for VLM perception tasks."""


# =================================== IMPORTS ==================================

# Standard Library
import io
import json
import logging

# Third Party
from google import genai
from google.genai import types
from PIL import Image
from pydantic import TypeAdapter

# Local
from tgenai_agent.vlm.types import LabeledPoint2D, ObjectFindingResult


# =================================== CONSTANTS ==================================

DETECT_COORDINATES_PROMPT = """\
Point to the following object in the image: {description}.
The answer should follow the json format:
[{{"point": <point>, "label": <label>}}].
The points are in [y, x] format normalized to 0-1000.
"""

OBJECT_FINDING_PROMPT = """
Find all objects in this image that match this description: '{description}'.

Provide a JSON response with two parts:
1. objects: An array of detected objects with associated center points
2. description: A natural language description of what you see, including spatial relationships

For the description, be creative but short (maximum two sentences) and include spatial context such as:
- Where objects are located relative to each other (e.g., next to, to the left of, in front of)
- Any interesting observations about the objects

Return the response in this exact JSON format (no code fencing, no markdown):
{{
  "objects": [{{"point": <point>, "label": <label>}}, ...],
  "description": "Your natural language description here"
}}

Rules:
- Points must be in [y, x] format, normalized to a 0-1000 range
- If multiple matching objects exist, include all of them
- Name objects by their unique characteristics when there are multiple
- Only include objects that match the description
- The description should be engaging and informative
"""

SYSTEM_INSTRUCTION = (
    "You are a precise visual perception system for robotics. "
    "Always respond with only the requested JSON format, no extra text."
)

Point2DListAdapter = TypeAdapter(list[LabeledPoint2D])


# ==================================== CLASSES ====================================

class GeminiRoboticsER:
    """Thin wrapper around the native Gemini API for Robotics-ER calls."""

    # ------------------------- INITIALIZATION -------------------------

    def __init__(self, api_key: str, model_id: str) -> None:
        self._logger = logging.getLogger(__name__)
        self._api_key = api_key
        self._model_id = model_id
        self._client = genai.Client(api_key=api_key) if api_key else None

    # ------------------------- PUBLIC METHODS -------------------------

    async def detect_object_coordinates(
        self,
        image_bytes: bytes,
        description: str,
    ) -> ObjectFindingResult:
        """Detect objects and return their normalized [y, x] coordinates plus a scene description."""
        if self._client is None:
            raise ValueError("GOOGLE_API_KEY is required for detect_object_coordinates")

        prompt = OBJECT_FINDING_PROMPT.format(description=description)
        self._logger.info("VLM prompt: %s", prompt)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        response = await self._client.aio.models.generate_content(
            model=self._model_id,
            contents=[image, f"{SYSTEM_INSTRUCTION}\n\n{prompt}"],
            config=types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        raw = response.text or '{"objects": [], "description": ""}'
        self._logger.info("Raw VLM response: %s", raw)

        cleaned = self._clean_json_response(raw)
        result = ObjectFindingResult.model_validate_json(cleaned)
        self._logger.info("Object finding result: %s", result)
        return result

    # ------------------------- INTERNAL HELPERS -------------------------

    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """Strip markdown code fences from model output if present."""
        text = raw.strip()
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if not lines:
            return text

        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        candidate = "\n".join(lines).strip()

        # Keep the original if fence trimming produced invalid JSON text.
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return text
