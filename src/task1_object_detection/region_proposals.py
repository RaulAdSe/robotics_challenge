"""Region proposal generation using Selective Search and blob detection."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import time

class RegionProposalGenerator:
    """Generate region proposals for object detection using multiple methods."""
    
    def __init__(self, method: str = "selective_search", min_size: int = 500, max_proposals: int = 1000):
        """
        Initialize region proposal generator.
        
        Args:
            method: "selective_search", "blob_detection", or "both"
            min_size: Minimum area for valid proposals
            max_proposals: Maximum number of proposals to return
        """
        self.method = method
        self.min_size = min_size
        self.max_proposals = max_proposals
        
    def generate_proposals(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Generate region proposals from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        if self.method == "selective_search":
            return self._selective_search_proposals(image)
        elif self.method == "blob_detection":
            return self._blob_detection_proposals(image)
        elif self.method == "both":
            ss_proposals = self._selective_search_proposals(image)
            blob_proposals = self._blob_detection_proposals(image)
            combined = ss_proposals + blob_proposals
            return self._filter_and_limit_proposals(combined)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _selective_search_proposals(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Generate proposals using OpenCV's Selective Search."""
        try:
            # Create Selective Search Segmentation Object
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(image)
            
            # Use fast mode for speed (can switch to quality mode)
            ss.switchToSelectiveSearchFast()
            
            # Run selective search segmentation
            rects = ss.process()
            
            # Convert to list of tuples and filter
            proposals = [(int(x), int(y), int(w), int(h)) for x, y, w, h in rects]
            return self._filter_and_limit_proposals(proposals)
            
        except Exception as e:
            print(f"Selective Search failed: {e}")
            print("Falling back to blob detection...")
            return self._blob_detection_proposals(image)
    
    def _blob_detection_proposals(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Generate proposals using blob detection and contour finding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Blob detector
        proposals = []
        
        # Simple blob detector
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)
        
        for kp in keypoints:
            x = int(kp.pt[0] - kp.size)
            y = int(kp.pt[1] - kp.size)
            w = h = int(kp.size * 2)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                proposals.append((x, y, w, h))
        
        # Method 2: Contour-based proposals
        contour_proposals = self._contour_based_proposals(gray)
        proposals.extend(contour_proposals)
        
        return self._filter_and_limit_proposals(proposals)
    
    def _contour_based_proposals(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Generate proposals based on contours."""
        proposals = []
        
        # Apply different thresholds to capture various object types
        thresholds = [127, 100, 150, 80, 180]
        
        for threshold in thresholds:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by area
                if area > self.min_size:
                    proposals.append((x, y, w, h))
        
        return proposals
    
    def _filter_and_limit_proposals(self, proposals: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Filter proposals by area and limit the number returned."""
        # Filter by minimum size
        filtered = [(x, y, w, h) for x, y, w, h in proposals 
                   if w * h >= self.min_size]
        
        # Remove duplicates and very similar proposals
        filtered = self._remove_duplicate_proposals(filtered)
        
        # Sort by area (larger first) and limit number
        filtered.sort(key=lambda box: box[2] * box[3], reverse=True)
        
        return filtered[:self.max_proposals]
    
    def _remove_duplicate_proposals(self, proposals: List[Tuple[int, int, int, int]], 
                                  iou_threshold: float = 0.8) -> List[Tuple[int, int, int, int]]:
        """Remove very similar proposals based on IoU."""
        if not proposals:
            return []
        
        # Sort by area
        proposals.sort(key=lambda box: box[2] * box[3], reverse=True)
        
        keep = []
        for box in proposals:
            should_keep = True
            for kept_box in keep:
                if self._calculate_iou(box, kept_box) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(box)
        
        return keep
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
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

def benchmark_proposal_methods(image: np.ndarray) -> dict:
    """Benchmark different proposal generation methods."""
    methods = ["selective_search", "blob_detection"]
    results = {}
    
    for method in methods:
        generator = RegionProposalGenerator(method=method)
        
        start_time = time.time()
        proposals = generator.generate_proposals(image)
        end_time = time.time()
        
        results[method] = {
            "proposals_count": len(proposals),
            "execution_time": end_time - start_time,
            "proposals": proposals[:10]  # Store first 10 for analysis
        }
        
        print(f"{method}: {len(proposals)} proposals in {end_time - start_time:.4f}s")
    
    return results

if __name__ == "__main__":
    # Test the region proposal generator
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.visualization import draw_bounding_boxes
    
    # Load test image
    test_image_path = "tests/test_images/test_objects_simple.jpg"
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        
        # Test different methods
        results = benchmark_proposal_methods(image)
        
        # Visualize results
        for method, result in results.items():
            proposals = result["proposals"]
            output = draw_bounding_boxes(image, proposals)
            output_path = f"debug_{method}_proposals.jpg"
            cv2.imwrite(output_path, output)
            print(f"Saved visualization: {output_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Run 'python tests/generate_test_images.py' first")