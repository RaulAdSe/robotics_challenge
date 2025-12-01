"""Gradient-based barcode detection without pre-trained models."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class BarcodeDetector:
    """Detect barcodes using gradient-based morphological operations."""
    
    def __init__(self, 
                 min_area: int = 500,
                 max_area: int = 500000,
                 aspect_ratio_range: Tuple[float, float] = (2.0, 15.0),
                 gradient_threshold: int = 225,
                 blur_kernel_size: int = 9,
                 morph_kernel_size: Tuple[int, int] = (21, 7)):
        """
        Initialize barcode detector.
        
        Args:
            min_area: Minimum area for valid barcode regions
            max_area: Maximum area for valid barcode regions  
            aspect_ratio_range: (min_ratio, max_ratio) for barcode rectangles
            gradient_threshold: Threshold for binary gradient image
            blur_kernel_size: Kernel size for Gaussian blur
            morph_kernel_size: (width, height) for morphological operations
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.gradient_threshold = gradient_threshold
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
    
    def detect_barcodes(self, image: np.ndarray, debug: bool = False) -> Tuple[List[np.ndarray], Dict]:
        """
        Detect barcode regions in an image.
        
        Args:
            image: Input image (BGR format)
            debug: Whether to return debug information and intermediate images
            
        Returns:
            Tuple of (barcode_regions, debug_info)
            - barcode_regions: List of corner arrays for detected barcodes
            - debug_info: Dictionary with intermediate processing results
        """
        debug_info = {
            "input_shape": image.shape,
            "processing_stages": {},
            "detected_count": 0
        }
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_info["processing_stages"]["grayscale"] = gray if debug else None
        
        # Step 2: Calculate gradients (emphasize vertical edges)
        gradient_img = self._calculate_barcode_gradients(gray)
        debug_info["processing_stages"]["gradients"] = gradient_img if debug else None
        
        # Step 3: Blur and threshold
        blurred = cv2.blur(gradient_img, (self.blur_kernel_size, self.blur_kernel_size))
        _, thresh = cv2.threshold(blurred, self.gradient_threshold, 255, cv2.THRESH_BINARY)
        debug_info["processing_stages"]["threshold"] = thresh if debug else None
        
        # Step 4: Morphological operations to connect bars
        morph_img = self._morphological_closing(thresh)
        debug_info["processing_stages"]["morphology"] = morph_img if debug else None
        
        # Step 5: Find and filter contours
        barcode_contours = self._find_barcode_contours(morph_img)
        debug_info["candidate_contours"] = len(barcode_contours)
        
        # Step 6: Extract oriented bounding boxes
        barcode_regions = []
        for contour in barcode_contours:
            # Get minimum area rectangle (handles rotation)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            barcode_regions.append(box)
        
        debug_info["detected_count"] = len(barcode_regions)
        
        return barcode_regions, debug_info
    
    def _calculate_barcode_gradients(self, gray: np.ndarray) -> np.ndarray:
        """
        Calculate gradients emphasizing vertical barcode patterns.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Gradient magnitude image emphasizing vertical edges
        """
        # Calculate gradients in X and Y directions
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        
        # Subtract Y gradients from X gradients
        # This emphasizes vertical lines (high X gradient) and removes horizontal lines
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        
        return gradient
    
    def _morphological_closing(self, thresh: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing to connect individual barcode bars.
        
        Args:
            thresh: Binary threshold image
            
        Returns:
            Morphologically processed image
        """
        # Create rectangular kernel - wider than tall to connect vertical bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        
        # Apply closing operation to fill gaps between bars
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def _find_barcode_contours(self, morph_img: np.ndarray) -> List[np.ndarray]:
        """
        Find and filter contours that could be barcodes.
        
        Args:
            morph_img: Morphologically processed image
            
        Returns:
            List of contours that pass barcode criteria
        """
        # Find contours
        contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle to check aspect ratio
            rect = cv2.minAreaRect(contour)
            (_, _), (width, height), _ = rect
            
            # Ensure width > height (barcodes are typically wider than tall)
            if width < height:
                width, height = height, width
            
            # Calculate aspect ratio
            if height > 0:
                aspect_ratio = width / height
                
                # Filter by aspect ratio
                if (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                    valid_contours.append(contour)
        
        return valid_contours
    
    def extract_barcode_region(self, image: np.ndarray, corners: np.ndarray, 
                             output_size: Tuple[int, int] = (300, 100)) -> np.ndarray:
        """
        Extract and deskew barcode region from image.
        
        Args:
            image: Input image
            corners: 4 corner points of the barcode region
            output_size: (width, height) of output rectified image
            
        Returns:
            Rectified barcode image
        """
        # Order corners: top-left, top-right, bottom-right, bottom-left
        ordered_corners = self._order_corners(corners)
        
        # Define destination rectangle
        width, height = output_size
        dst_corners = np.array([
            [0, 0],
            [width - 1, 0], 
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation
        transform_matrix = cv2.getPerspectiveTransform(
            ordered_corners.astype(np.float32), 
            dst_corners
        )
        
        # Apply transformation
        rectified = cv2.warpPerspective(image, transform_matrix, output_size)
        
        return rectified
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered corner points
        """
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = []
        for corner in corners:
            angle = math.atan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)
        
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        
        # Reorder: start from top-left and go clockwise
        # This assumes the first sorted angle corresponds to top-left region
        ordered = corners[sorted_indices]
        
        return ordered
    
    def analyze_barcode_candidates(self, image: np.ndarray) -> Dict:
        """
        Analyze detected barcode candidates for debugging.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with analysis results
        """
        regions, debug_info = self.detect_barcodes(image, debug=True)
        
        analysis = {
            "total_candidates": debug_info["detected_count"],
            "processing_stages": list(debug_info["processing_stages"].keys()),
            "regions_analysis": []
        }
        
        for i, region in enumerate(regions):
            rect = cv2.minAreaRect(region)
            (center_x, center_y), (width, height), angle = rect
            
            # Ensure consistent width/height orientation
            if width < height:
                width, height = height, width
                angle += 90
            
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            analysis["regions_analysis"].append({
                "id": i,
                "center": (center_x, center_y),
                "size": (width, height),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "rotation_angle": angle
            })
        
        return analysis

def visualize_barcode_detection(image: np.ndarray, 
                               detector: BarcodeDetector,
                               output_path: str = "barcode_detection_debug.jpg") -> None:
    """
    Visualize barcode detection process with debug stages.
    
    Args:
        image: Input image
        detector: BarcodeDetector instance
        output_path: Path to save visualization
    """
    try:
        from ..utils.visualization import create_debug_grid
    except ImportError:
        # Fallback for testing without package structure
        print(f"Visualization not available - would save to {output_path}")
        return
    
    # Run detection with debug info
    regions, debug_info = detector.detect_barcodes(image, debug=True)
    
    # Prepare images for grid
    debug_images = []
    titles = []
    
    # Original image with detections
    result_img = image.copy()
    for region in regions:
        cv2.drawContours(result_img, [region], -1, (0, 255, 0), 2)
    debug_images.append(result_img)
    titles.append("Detected Barcodes")
    
    # Add processing stages
    for stage_name, stage_img in debug_info["processing_stages"].items():
        if stage_img is not None:
            debug_images.append(stage_img)
            titles.append(stage_name.title())
    
    # Create debug grid
    grid = create_debug_grid(debug_images, titles)
    
    # Save visualization
    cv2.imwrite(output_path, grid)
    print(f"Barcode detection visualization saved: {output_path}")

if __name__ == "__main__":
    # Test barcode detection
    import sys
    from pathlib import Path
    
    # Test with available images
    test_images = [
        "Images/input_1.jpg", 
        "Images/input_2.jpg",
        "Images/input_3.jpg"
    ]
    
    detector = BarcodeDetector()
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\nTesting with {img_path}")
            image = cv2.imread(img_path)
            
            if image is not None:
                # Analyze candidates
                analysis = detector.analyze_barcode_candidates(image)
                print(f"Found {analysis['total_candidates']} barcode candidates")
                
                for region_info in analysis["regions_analysis"]:
                    print(f"  Region {region_info['id']}: "
                          f"size={region_info['size']}, "
                          f"aspect_ratio={region_info['aspect_ratio']:.2f}, "
                          f"angle={region_info['rotation_angle']:.1f}°")
                
                # Save visualization for first image
                if img_path == test_images[0]:
                    visualize_barcode_detection(image, detector, f"debug_{Path(img_path).stem}.jpg")
            else:
                print(f"Could not load {img_path}")
        else:
            print(f"File not found: {img_path}")