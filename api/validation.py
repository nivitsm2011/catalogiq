"""Input validation: uploaded photos (type, size, content) and a common error type."""

from __future__ import annotations

import io

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from catalogiq.config import get_api_config

_UPLOAD = get_api_config()["upload"]
MAX_BYTES = int(_UPLOAD["max_bytes"])  # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_SIDE = int(_UPLOAD["min_side"])
MAX_PIXELS = int(_UPLOAD["max_pixels"])


class ApiError(Exception):
    """An error the client can act on: HTTP status, a stable code and a readable message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


async def read_image(upload: UploadFile | None, field: str = "file") -> Image.Image:
    """Read and validate an uploaded photo.

    Raises:
        ApiError: 400 if missing or empty, 413 if over 5 MB or too many pixels, 415 if not a
            JPEG, PNG or WebP, 422 if the bytes are not a readable image or are too small.
    """
    if upload is None:
        raise ApiError(
            400, "missing_file", f"Send a photo in the '{field}' field (JPEG, PNG or WebP)."
        )
    if upload.content_type not in ALLOWED_TYPES:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"'{upload.filename}' is {upload.content_type or 'of unknown type'}; "
            "only JPEG, PNG and WebP photos are accepted.",
        )
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ApiError(
            413,
            "file_too_large",
            f"The photo is larger than the {MAX_BYTES // 1024 // 1024} MB limit.",
        )
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt, size = probe.format, probe.size
        if fmt not in ALLOWED_FORMATS:
            raise ApiError(
                415, "unsupported_media_type", f"The file content is {fmt}, not JPEG, PNG or WebP."
            )
        if size[0] * size[1] > MAX_PIXELS:
            raise ApiError(
                413,
                "image_too_large",
                "The photo has too many pixels; resize it below 40 megapixels.",
            )
        probe.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ApiError(
            422, "invalid_image", f"The file is not a readable image ({exc.__class__.__name__})."
        ) from exc
    if min(image.size) < MIN_SIDE:
        raise ApiError(
            422,
            "image_too_small",
            f"The photo is {image.size[0]}x{image.size[1]}; each side needs {MIN_SIDE}+ px.",
        )
    return image
