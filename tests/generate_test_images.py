"""Generate synthetic test images for development and testing."""

import cv2
import numpy as np
from pathlib import Path

def create_simple_test_image():
    """Create a simple test image with basic shapes."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some simple objects
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (400, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img, (300, 300), (450, 400), (0, 0, 255), -1)  # Red rectangle
    
    return img

def create_barcode_test_image():
    """Create a simple synthetic barcode-like pattern for testing."""
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    
    # Create vertical bars pattern (simplified barcode-like)
    x_start = 150
    bar_width = 3
    for i in range(50):
        if i % 2 == 0:  # Draw every other bar
            x1 = x_start + i * bar_width
            x2 = x1 + bar_width
            cv2.rectangle(img, (x1, 80), (x2, 220), (0, 0, 0), -1)
    
    return img

def generate_test_images():
    """Generate test images for development."""
    test_dir = Path("tests/test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Generate simple test image
    simple_img = create_simple_test_image()
    cv2.imwrite(str(test_dir / "test_objects_simple.jpg"), simple_img)
    
    # Generate barcode test image  
    barcode_img = create_barcode_test_image()
    cv2.imwrite(str(test_dir / "test_barcode_simple.jpg"), barcode_img)
    
    print(f"Generated test images in {test_dir}")

if __name__ == "__main__":
    generate_test_images()