"""Validation of untrusted image uploads.

Responsibility: turn an `UploadFile` into bytes that are safe to hand to
Pillow/CLIP, or raise an `HTTPException` explaining why not. The
`Content-Type` a client declares is just a claim, so this checks, in
order: declared type is allowed, size is under `MAX_UPLOAD_BYTES`
(read in chunks, never trusting `Content-Length`), the file is
non-empty, the *actual* format (from the bytes) is allowed and agrees
with the declared type, it decodes as an image, and its pixel count is
under `MAX_IMAGE_PIXELS` (decompression-bomb guard).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES

_FORMAT_TO_CONTENT_TYPE = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    content_type: str  # derived from the bytes, not from the client's claim


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _too_large(detail: str) -> HTTPException:
    return HTTPException(status_code=413, detail=detail)


async def read_validated_image(
    file: UploadFile,
    allowed_content_types: set[str],
    *,
    label: str | None = None,
    max_bytes: int | None = None,
) -> ValidatedImage:
    """Read and validate one uploaded image.

    Args:
        file: The multipart upload.
        allowed_content_types: e.g. `{"image/jpeg", "image/png"}`; applies to
            both the declared type and the type sniffed from the bytes.
        label: How to refer to the file in error messages.
        max_bytes: Size cap for this file (default `MAX_UPLOAD_BYTES`).

    Raises:
        HTTPException 400: Unsupported/mismatched type, empty, corrupt, or too many pixels.
        HTTPException 413: Larger than `max_bytes`.
    """
    names = sorted(t.removeprefix("image/").upper() for t in allowed_content_types)
    allowed_text = " or ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + ", or " + names[-1]
    max_bytes = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    prefix = f"{label}: " if label else ""
    subject = label or "Uploaded file"
    if file.content_type not in allowed_content_types:
        raise _bad_request(f"{prefix}{'unsupported' if label else 'Unsupported'} file type '{file.content_type}'. Use {allowed_text}.")

    if file.size is not None and file.size > max_bytes:
        raise _too_large(f"{subject} is too large (max {max_bytes} bytes).")

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise _too_large(f"{subject} is too large (max {max_bytes} bytes).")
        chunks.append(chunk)
    data = b"".join(chunks)

    if not data:
        raise _bad_request(f"{subject} is empty.")

    try:
        with Image.open(io.BytesIO(data)) as image:  # lazy: reads the header only
            detected_type = _FORMAT_TO_CONTENT_TYPE.get(image.format or "")
            width, height = image.size
            if detected_type is None or detected_type not in allowed_content_types:
                raise _bad_request(f"{prefix}File contents are not a supported image ({allowed_text}).")
            if detected_type != file.content_type:
                raise _bad_request(
                    f"{prefix}Declared type '{file.content_type}' does not match contents ('{detected_type}')."
                )
            if width * height > MAX_IMAGE_PIXELS:
                raise _bad_request(f"{prefix}Image dimensions {width}x{height} exceed the {MAX_IMAGE_PIXELS}-pixel limit.")
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise _bad_request(f"{subject} is not a valid image.") from exc

    return ValidatedImage(data=data, content_type=detected_type)
