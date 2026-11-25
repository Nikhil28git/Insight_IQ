from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, Any

from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv(override=True)

# Single source of truth for the Gemini model used across the app.
# Override by setting GEMINI_MODEL=... in .env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"

PROMPT = """
Describe this image for retrieval and question-answering purposes.

Capture:

- Visible text
- Labels
- Headings
- Table contents
- Diagram relationships
- Dimensions and measurements
- Technical information
- Product names
- Component names
- Engineering details
- Important entities
- Document meaning

Special handling:

- If this is a table, extract the table contents.
- If this is a diagram, describe the relationships shown.
- If this is a technical drawing or CAD model, describe components and functionality.
- If this is a scanned document, extract meaningful information.

Return concise retrieval-oriented plain text.

Do not describe colors, aesthetics, or visual style unless they are technically important.
"""


def _cache_path(image_path: str) -> Path:
    return Path(f"{image_path}.txt")


def describe_image_with_gemini(
    image_path: str,
    api_key: str | None = None,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Tuple[str, Dict[str, Any]]:

    cache_file = _cache_path(image_path)

    # --------------------------------------------------
    # Cache
    # --------------------------------------------------
    if use_cache and cache_file.exists() and not force_refresh:
        try:
            cached_text = cache_file.read_text(
                encoding="utf-8"
            ).strip()

            if cached_text:
                return cached_text, {"cached": True}

        except Exception:
            pass

    # --------------------------------------------------
    # API Key
    # --------------------------------------------------
    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not key:
        return (
            "[IMAGE_DESCRIPTION_PENDING] (No API key found)",
            {"error": "no_api_key"},
        )

    # --------------------------------------------------
    # Image Exists
    # --------------------------------------------------
    if not os.path.exists(image_path):
        return (
            f"[IMAGE_DESCRIPTION_PENDING] (Image not found: {image_path})",
            {"error": "image_not_found"},
        )

    try:
        print(f"Processing image: {image_path}")

        client = genai.Client(api_key=key)

        img = Image.open(image_path)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                PROMPT,
                img,
            ],
        )

        description = response.text.strip()

        if not description:
            return (
                "[IMAGE_DESCRIPTION_PENDING] (Empty Gemini response)",
                {"error": "empty_response"},
            )

        # --------------------------------------------------
        # Save Cache
        # --------------------------------------------------
        try:
            cache_file.write_text(
                description,
                encoding="utf-8"
            )
        except Exception:
            pass

        print(description[:300])

        return description, {
            "cached": False,
            "model": GEMINI_MODEL,
        }

    except Exception as e:
        return (
            f"[IMAGE_DESCRIPTION_PENDING] ({str(e)})",
            {"error": str(e)},
        )