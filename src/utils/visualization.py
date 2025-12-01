"""Visualization utilities for drawing bounding boxes, arrows, and debug outputs."""

import cv2
import numpy as np
from typing import List, Tuple, Optional

def draw_bounding_boxes(image: np.ndarray, 
                       boxes: List[Tuple[int, int, int, int]], 
                       labels: Optional[List[str]] = None,
                       confidences: Optional[List[float]] = None,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on image with optional labels and confidence scores."""
    result = image.copy()
    
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        if labels is not None and i < len(labels):
            label_text = labels[i]
            if confidences is not None and i < len(confidences):
                label_text += f" ({confidences[i]:.2f})"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(result, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result

def draw_arrow_3d(image: np.ndarray, 
                 center: Tuple[int, int],
                 direction: Tuple[int, int],
                 length: int = 100,
                 color: Tuple[int, int, int] = (0, 0, 255),
                 thickness: int = 3) -> np.ndarray:
    """Draw a 3D arrow representing surface normal vector."""
    result = image.copy()
    
    end_x = center[0] + int(direction[0] * length)
    end_y = center[1] + int(direction[1] * length)
    
    # Draw arrow line
    cv2.arrowedLine(result, center, (end_x, end_y), color, thickness)
    
    return result

def create_debug_grid(images: List[np.ndarray], 
                     titles: List[str],
                     grid_size: Tuple[int, int] = None) -> np.ndarray:
    """Create a grid of images for debugging visualization."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_images = len(images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    # Resize all images to same size
    target_h, target_w = 300, 400
    resized_images = []
    
    for i, img in enumerate(images):
        if len(img.shape) == 3:
            resized = cv2.resize(img, (target_w, target_h))
        else:
            resized = cv2.resize(img, (target_w, target_h))
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        # Add title
        title = titles[i] if i < len(titles) else f"Image {i}"
        cv2.putText(resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        resized_images.append(resized)
    
    # Create grid
    grid_h = rows * target_h
    grid_w = cols * target_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols
        y_start = row * target_h
        y_end = y_start + target_h
        x_start = col * target_w
        x_end = x_start + target_w
        
        grid[y_start:y_end, x_start:x_end] = img
    
    return grid

def save_debug_output(image: np.ndarray, 
                     output_path: str,
                     title: str = "Debug Output") -> None:
    """Save debug visualization with timestamp."""
    from datetime import datetime
    
    # Add timestamp to image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_text = f"{title} - {timestamp}"
    
    output = image.copy()
    cv2.putText(output, debug_text, (10, image.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output, debug_text, (10, image.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, output)
    print(f"Debug output saved: {output_path}")