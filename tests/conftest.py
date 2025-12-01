"""Pytest configuration and fixtures."""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (400, 150), 50, (0, 255, 0), -1)
    return img

@pytest.fixture
def sample_barcode_image():
    """Create a sample barcode-like image."""
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    x_start = 150
    bar_width = 3
    for i in range(50):
        if i % 2 == 0:
            x1 = x_start + i * bar_width
            x2 = x1 + bar_width
            cv2.rectangle(img, (x1, 80), (x2, 220), (0, 0, 0), -1)
    return img

@pytest.fixture
def test_images_dir():
    """Get the test images directory path."""
    return Path(__file__).parent / "test_images"

@pytest.fixture(scope="session")
def ensure_test_images():
    """Ensure test images exist before running tests."""
    from tests.generate_test_images import generate_test_images
    generate_test_images()
    yield
    # Cleanup if needed