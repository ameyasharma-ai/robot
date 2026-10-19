"""Unit tests for the Image Studio Gallery HUD API and file management endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from robot.api import create_app


@pytest.fixture
def test_gallery_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture providing a temporary directory for data/images artwork storage.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch utility.

    Returns:
        Path to the isolated temporary root directory.
    """
    monkeypatch.chdir(tmp_path)
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir


def _create_dummy_image(path: Path, width: int = 256, height: int = 256) -> None:
    """Create a valid PNG image file on disk for test validation.

    Args:
        path: Target file path to write.
        width: Pixel width of image.
        height: Pixel height of image.
    """
    im = Image.new("RGB", (width, height), color=(73, 109, 137))
    im.save(path, format="PNG")


def test_gallery_empty_directory(test_gallery_env: Path) -> None:
    """Verify GET /api/images/gallery returns an empty list when no images exist.

    Args:
        test_gallery_env: Isolated images directory fixture.
    """
    engine = MagicMock()
    app = create_app(engine)
    client = TestClient(app)

    response = client.get("/api/images/gallery")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total"] == 0
    assert data["images"] == []


def test_gallery_populated_with_job_metadata(test_gallery_env: Path) -> None:
    """Verify GET /api/images/gallery returns rich metadata matching in-memory jobs.

    Args:
        test_gallery_env: Isolated images directory fixture.
    """
    engine = MagicMock()
    # Mock image_service.list_jobs
    job_mock = {
        "job_id": "img_job_12345",
        "prompt": "A majestic cyberpunk neon robot cat",
        "model": "sdxl-turbo",
        "duration_s": 2.45,
        "output_path": str(test_gallery_env / "artwork_cyber_cat.png"),
    }
    engine.image_service.list_jobs.return_value = [job_mock]

    # Create dummy images
    _create_dummy_image(test_gallery_env / "artwork_cyber_cat.png", width=512, height=512)
    _create_dummy_image(test_gallery_env / "img_test_flux_01.png", width=1024, height=768)

    app = create_app(engine)
    client = TestClient(app)

    response = client.get("/api/images/gallery")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total"] == 2
    assert len(data["images"]) == 2

    # Verify matched metadata from job
    cyber_cat = next(img for img in data["images"] if img["filename"] == "artwork_cyber_cat.png")
    assert cyber_cat["prompt"] == "A majestic cyberpunk neon robot cat"
    assert cyber_cat["model"] == "sdxl-turbo"
    assert cyber_cat["width"] == 512
    assert cyber_cat["height"] == 512
    assert cyber_cat["duration_s"] == 2.45
    assert cyber_cat["url"] == "/data/images/artwork_cyber_cat.png"

    # Verify fallback metadata for second image
    flux_img = next(img for img in data["images"] if img["filename"] == "img_test_flux_01.png")
    assert "flux-schnell" in flux_img["model"]
    assert flux_img["width"] == 1024
    assert flux_img["height"] == 768


def test_delete_gallery_image_success(test_gallery_env: Path) -> None:
    """Verify DELETE /api/images/gallery/{filename} cleanly removes valid image.

    Args:
        test_gallery_env: Isolated images directory fixture.
    """
    engine = MagicMock()
    target_file = test_gallery_env / "to_delete.png"
    _create_dummy_image(target_file)
    assert target_file.exists()

    app = create_app(engine)
    client = TestClient(app)

    response = client.delete("/api/images/gallery/to_delete.png")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["deleted"] is True
    assert not target_file.exists()


def test_delete_gallery_image_path_traversal_protection(test_gallery_env: Path) -> None:
    """Verify DELETE /api/images/gallery rejects traversal attempts and non-existent files.

    Args:
        test_gallery_env: Isolated images directory fixture.
    """
    engine = MagicMock()
    app = create_app(engine)
    client = TestClient(app)

    # 1. Path traversal attempt with ..
    resp = client.delete("/api/images/gallery/..%2Fetc%2Fpasswd")
    # FastAPI route matching or endpoint regex validation rejects this
    assert resp.status_code in (404, 200)
    if resp.status_code == 200:
        assert resp.json()["status"] == "error"

    # 2. Non-existent file
    resp = client.delete("/api/images/gallery/non_existent.png")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
    assert "not found" in resp.json()["message"].lower()

    # 3. Invalid characters
    resp = client.delete("/api/images/gallery/bad*name?.png")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
    assert "invalid" in resp.json()["message"].lower()
