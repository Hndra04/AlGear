"""Utility functions for image/video encoding and file validation."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 100


def validate_image_file(filename: str, size_bytes: int) -> str | None:
    """Return error message if invalid, None if ok."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return f"Unsupported image format '{ext}'. Allowed: {ALLOWED_IMAGE_EXTS}"
    if size_bytes > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        return f"Image too large ({size_bytes / 1024 / 1024:.1f}MB). Max: {MAX_IMAGE_SIZE_MB}MB"
    return None


def validate_video_file(filename: str, size_bytes: int) -> str | None:
    """Return error message if invalid, None if ok."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return f"Unsupported video format '{ext}'. Allowed: {ALLOWED_VIDEO_EXTS}"
    if size_bytes > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        return f"Video too large ({size_bytes / 1024 / 1024:.1f}MB). Max: {MAX_VIDEO_SIZE_MB}MB"
    return None


def encode_image_to_base64(frame: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR numpy frame to base64 data URI string."""
    success, buffer = cv2.imencode(fmt, frame)
    if not success:
        raise ValueError(f"Failed to encode image with format {fmt}")
    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else f"image/{fmt.lstrip('.')}"
    return f"data:{mime};base64,{b64}"


def encode_video_to_base64(video_path: Path) -> str:
    """Read an entire video file and encode to base64 data URI."""
    data = video_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suffix = video_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    mime = mime_map.get(suffix, "video/mp4")
    return f"data:{mime};base64,{b64}"


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image. File may be corrupted.")
    return frame
