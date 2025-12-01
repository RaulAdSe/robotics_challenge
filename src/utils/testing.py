"""Testing utilities for validation and benchmarking."""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from contextlib import contextmanager

class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name}: {elapsed:.4f}s")

class TestImageLoader:
    """Load and manage test images for validation."""
    
    def __init__(self, test_dir: str = "tests/test_images"):
        self.test_dir = Path(test_dir)
        
    def load_task1_images(self) -> List[Tuple[np.ndarray, str]]:
        """Load test images for Task 1 (object detection)."""
        images = []
        pattern = "test_objects_*.jpg"
        
        for img_path in self.test_dir.glob(pattern):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((img, img_path.stem))
        
        return images
    
    def load_task2_images(self) -> List[Tuple[np.ndarray, str]]:
        """Load test images for Task 2 (barcode detection)."""
        images = []
        pattern = "test_barcode_*.jpg"
        
        for img_path in self.test_dir.glob(pattern):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((img, img_path.stem))
        
        return images
    
    def get_test_image(self, name: str) -> Optional[np.ndarray]:
        """Get a specific test image by name."""
        img_path = self.test_dir / f"{name}.jpg"
        if img_path.exists():
            return cv2.imread(str(img_path))
        return None

class ValidationMetrics:
    """Calculate validation metrics for object detection."""
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def evaluate_detections(predicted_boxes: List[Tuple[int, int, int, int]],
                          ground_truth_boxes: List[Tuple[int, int, int, int]],
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate detection performance against ground truth."""
        if not ground_truth_boxes:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        true_positives = 0
        matched_gt = set()
        
        for pred_box in predicted_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(ground_truth_boxes):
                if gt_idx in matched_gt:
                    continue
                    
                iou = ValidationMetrics.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
        
        false_positives = len(predicted_boxes) - true_positives
        false_negatives = len(ground_truth_boxes) - true_positives
        
        precision = true_positives / len(predicted_boxes) if predicted_boxes else 0.0
        recall = true_positives / len(ground_truth_boxes)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

def run_pipeline_tests(pipeline_func, test_images: List[Tuple[np.ndarray, str]], 
                      test_name: str) -> Dict[str, Any]:
    """Run a pipeline function on test images and collect results."""
    results = {
        "test_name": test_name,
        "total_images": len(test_images),
        "successful": 0,
        "failed": 0,
        "errors": [],
        "execution_times": [],
        "outputs": []
    }
    
    for img, img_name in test_images:
        try:
            with PerformanceTimer(f"Processing {img_name}") as timer:
                output = pipeline_func(img)
            
            results["successful"] += 1
            results["execution_times"].append(timer.end_time - timer.start_time)
            results["outputs"].append((img_name, output))
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{img_name}: {str(e)}")
            print(f"Error processing {img_name}: {e}")
    
    # Calculate statistics
    if results["execution_times"]:
        times = results["execution_times"]
        results["avg_time"] = np.mean(times)
        results["std_time"] = np.std(times)
        results["min_time"] = np.min(times)
        results["max_time"] = np.max(times)
    
    return results

@contextmanager
def temporary_debug_mode():
    """Context manager to enable debug outputs temporarily."""
    print("=== DEBUG MODE ENABLED ===")
    try:
        yield
    finally:
        print("=== DEBUG MODE DISABLED ===")