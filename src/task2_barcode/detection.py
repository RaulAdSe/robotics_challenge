"""Gradient-based barcode detection without pre-trained models.

This module detects Code128 barcodes using classical computer vision techniques:
- Gradient analysis (Scharr operators)
- Local variance for high-contrast regions
- Morphological operations
- Shape-based filtering

The detector can work on full images or constrained to object regions from Task 1.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math


class BarcodeDetector:
    """Detect barcodes using gradient-based morphological operations.

    This detector finds barcode regions by:
    1. Computing directional gradients (horizontal/vertical)
    2. Identifying high-variance (high-contrast) regions
    3. Using morphological closing to connect barcode bars
    4. Filtering candidates by shape metrics (aspect ratio, solidity, extent)
    """

    def __init__(self,
                 grad_thresh: float = 1.3,
                 var_thresh: int = 25,
                 min_solidity: float = 0.55,
                 min_extent: float = 0.35,
                 kernel_w: int = 33,
                 kernel_h: int = 7,
                 min_aspect: float = 2.0,
                 max_aspect: float = 12.0,
                 min_area_pct: float = 0.0003,
                 max_area_pct: float = 0.025):
        """
        Initialize barcode detector with optimized parameters.

        Args:
            grad_thresh: Gradient ratio threshold (dominant/cross direction)
            var_thresh: Variance threshold for high-contrast regions
            min_solidity: Minimum solidity (area / convex hull area)
            min_extent: Minimum extent (area / bounding rect area)
            kernel_w: Morphological kernel width
            kernel_h: Morphological kernel height
            min_aspect: Minimum aspect ratio (width/height)
            max_aspect: Maximum aspect ratio
            min_area_pct: Minimum area as percentage of image
            max_area_pct: Maximum area as percentage of image
        """
        self.grad_thresh = grad_thresh
        self.var_thresh = var_thresh
        self.min_solidity = min_solidity
        self.min_extent = min_extent
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct

    def detect_barcodes(self, image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None,
                        debug: bool = False,
                        use_absolute_area: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Detect barcode regions in an image or region of interest.

        Args:
            image: Input image (BGR format)
            roi: Optional region of interest as (x, y, width, height)
            debug: Whether to return debug information
            use_absolute_area: If True and roi is specified, use full image
                              area for percentage thresholds (prevents missing
                              small barcodes in small ROIs)

        Returns:
            Tuple of (barcode_detections, debug_info)
            - barcode_detections: List of dicts with 'box', 'center', 'size', etc.
            - debug_info: Dictionary with intermediate processing results
        """
        debug_info = {
            "input_shape": image.shape,
            "roi": roi,
            "processing_stages": {},
            "detected_count": 0
        }

        # Store full image area for threshold calculation
        full_img_area = image.shape[0] * image.shape[1]

        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi
            # Ensure ROI is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            if w <= 0 or h <= 0:
                return [], debug_info

            working_image = image[y:y+h, x:x+w].copy()
            roi_offset = (x, y)
        else:
            working_image = image
            roi_offset = (0, 0)

        img_h, img_w = working_image.shape[:2]
        # Use full image area for thresholds if specified (for ROI detection)
        img_area = full_img_area if (roi is not None and use_absolute_area) else img_h * img_w

        # Convert to grayscale
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if debug:
            debug_info["processing_stages"]["grayscale"] = gray.copy()

        # Compute gradients using Scharr (more accurate than Sobel)
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        grad_x_abs = np.abs(grad_x)
        grad_y_abs = np.abs(grad_y)

        # Compute local variance for high-contrast regions
        k = 15
        mean = cv2.blur(gray.astype(np.float32), (k, k))
        sq_mean = cv2.blur((gray.astype(np.float32))**2, (k, k))
        variance = np.clip(sq_mean - mean**2, 0, None)
        variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, high_var = cv2.threshold(variance_norm, self.var_thresh, 255, cv2.THRESH_BINARY)

        if debug:
            debug_info["processing_stages"]["variance"] = variance_norm.copy()
            debug_info["processing_stages"]["high_variance_mask"] = high_var.copy()

        # Calculate area thresholds
        min_area = int(img_area * self.min_area_pct)
        max_area = int(img_area * self.max_area_pct)

        candidates = []

        # Detect both horizontal and vertical barcodes
        orientations = [
            ('horizontal', grad_x_abs, grad_y_abs, (self.kernel_w, self.kernel_h)),
            ('vertical', grad_y_abs, grad_x_abs, (self.kernel_h, self.kernel_w)),
        ]

        for orientation, grad_main, grad_cross, kernel_size in orientations:
            # Create mask where dominant gradient direction exceeds threshold
            mask = ((grad_main > grad_cross * self.grad_thresh) &
                   (high_var > 0)).astype(np.uint8) * 255

            # Morphological closing to connect bars
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Clean up noise
            closed = cv2.erode(closed, None, iterations=2)
            closed = cv2.dilate(closed, None, iterations=3)

            if debug:
                debug_info["processing_stages"][f"mask_{orientation}"] = closed.copy()

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                candidate = self._validate_contour(contour, min_area, max_area, orientation)
                if candidate is not None:
                    # Adjust coordinates for ROI offset
                    candidate['box'] = candidate['box'] + np.array([roi_offset[0], roi_offset[1]])
                    candidate['center'] = (
                        candidate['center'][0] + roi_offset[0],
                        candidate['center'][1] + roi_offset[1]
                    )
                    candidates.append(candidate)

        # Apply non-maximum suppression
        detections = self._nms(candidates)

        debug_info["detected_count"] = len(detections)
        debug_info["candidates_before_nms"] = len(candidates)

        return detections, debug_info

    def detect_in_object_regions(self, image: np.ndarray,
                                  object_detections: List[Dict],
                                  object_names: Union[str, List[str]] = "all",
                                  expand_ratio: float = 0.1,
                                  one_barcode_per_object: bool = True) -> List[Dict]:
        """
        Detect barcodes within detected object regions from Task 1.

        Args:
            image: Input image (BGR format)
            object_detections: List of object detections from Task 1
                Each detection should have 'name' and 'bbox' (x, y, w, h)
            object_names: "all" or list of object names to process
            expand_ratio: How much to expand object bounding boxes (0.1 = 10%)
            one_barcode_per_object: If True, keep only the best barcode per object

        Returns:
            List of barcode detections, each with 'object_name' field
        """
        all_detections = []

        # Filter objects by name
        if object_names == "all":
            objects_to_process = object_detections
        else:
            if isinstance(object_names, str):
                object_names = [object_names]
            objects_to_process = [
                obj for obj in object_detections
                if obj.get('name', '').lower() in [n.lower() for n in object_names]
            ]

        for obj in objects_to_process:
            obj_name = obj.get('name', 'unknown')
            bbox = obj.get('bbox')

            if bbox is None:
                continue

            x, y, w, h = bbox

            # Expand bounding box slightly
            expand_x = int(w * expand_ratio)
            expand_y = int(h * expand_ratio)

            expanded_roi = (
                max(0, x - expand_x),
                max(0, y - expand_y),
                min(w + 2 * expand_x, image.shape[1] - max(0, x - expand_x)),
                min(h + 2 * expand_y, image.shape[0] - max(0, y - expand_y))
            )

            # Detect barcodes in this object's region
            # Use absolute area thresholds to handle small objects
            detections, _ = self.detect_barcodes(image, roi=expanded_roi, use_absolute_area=True)

            if not detections:
                continue

            if one_barcode_per_object:
                # Keep only the best barcode (largest area, highest solidity)
                best_det = max(detections, key=lambda d: d['area'] * d['solidity'])
                best_det['object_name'] = obj_name
                best_det['object_bbox'] = bbox
                all_detections.append(best_det)
            else:
                # Add all detections
                for det in detections:
                    det['object_name'] = obj_name
                    det['object_bbox'] = bbox
                    all_detections.append(det)

        # Final NMS across all objects (in case of overlapping regions)
        return self._nms(all_detections)

    def _validate_contour(self, contour: np.ndarray,
                          min_area: int, max_area: int,
                          orientation: str) -> Optional[Dict]:
        """
        Validate a contour as a potential barcode.

        Args:
            contour: OpenCV contour
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            orientation: 'horizontal' or 'vertical'

        Returns:
            Detection dict if valid, None otherwise
        """
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            return None

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)

        (cx, cy), (w, h), angle = rect

        # Ensure width > height
        if w < h:
            w, h = h, w
            angle = angle + 90 if angle < 0 else angle - 90

        if h == 0:
            return None

        aspect = w / h

        if aspect < self.min_aspect or aspect > self.max_aspect:
            return None

        # Shape metrics
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, bw, bh = cv2.boundingRect(contour)
        extent = area / (bw * bh) if bw * bh > 0 else 0

        if solidity < self.min_solidity or extent < self.min_extent:
            return None

        return {
            'box': box,
            'center': (cx, cy),
            'size': (w, h),
            'angle': angle,
            'area': area,
            'aspect': aspect,
            'solidity': solidity,
            'extent': extent,
            'orientation': orientation,
            'contour': contour
        }

    def _nms(self, candidates: List[Dict],
             distance_threshold_ratio: float = 0.5) -> List[Dict]:
        """
        Non-maximum suppression to remove duplicate detections.

        Args:
            candidates: List of candidate detections
            distance_threshold_ratio: Threshold as ratio of smaller dimension

        Returns:
            Filtered list of detections
        """
        if not candidates:
            return []

        # Sort by area (larger first)
        candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)

        kept = []
        for cand in candidates:
            is_duplicate = False
            for kept_cand in kept:
                dx = cand['center'][0] - kept_cand['center'][0]
                dy = cand['center'][1] - kept_cand['center'][1]
                dist = np.sqrt(dx*dx + dy*dy)

                min_dim = min(cand['size'][1], kept_cand['size'][1])
                if dist < min_dim * distance_threshold_ratio:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(cand)

        return kept

    def extract_barcode_region(self, image: np.ndarray,
                               detection: Dict,
                               output_size: Tuple[int, int] = (300, 100)) -> np.ndarray:
        """
        Extract and deskew barcode region from image.

        Args:
            image: Input image
            detection: Barcode detection dict with 'box' key
            output_size: (width, height) of output rectified image

        Returns:
            Rectified barcode image
        """
        corners = detection['box']
        ordered_corners = self._order_corners(corners)

        width, height = output_size
        dst_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(
            ordered_corners.astype(np.float32),
            dst_corners
        )

        rectified = cv2.warpPerspective(image, transform_matrix, output_size)

        return rectified

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners: top-left, top-right, bottom-right, bottom-left.
        """
        center = np.mean(corners, axis=0)

        angles = []
        for corner in corners:
            angle = math.atan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)

        sorted_indices = np.argsort(angles)
        ordered = corners[sorted_indices]

        return ordered


def test_detection_on_images():
    """Test barcode detection on sample images."""
    import os

    image_targets = {
        'Images/input_1.jpg': 5,
        'Images/input_2.jpg': 5,
        'Images/input_3.jpg': 5,
        'Images/input_4.jpg': 5,
        'Images/input_5.jpg': 3,
        'Images/input_6.jpg': 3,
        'Images/input_7.jpg': 6,
        'Images/input_8.jpg': 6,
    }

    detector = BarcodeDetector()

    print("Testing barcode detection:")
    total_correct = 0

    for img_path, target in image_targets.items():
        if not os.path.exists(img_path):
            print(f"  {img_path}: FILE NOT FOUND")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  {img_path}: COULD NOT LOAD")
            continue

        detections, debug_info = detector.detect_barcodes(image)

        status = '✓' if len(detections) == target else '✗'
        diff = len(detections) - target
        if len(detections) == target:
            total_correct += 1

        print(f"  {img_path}: found {len(detections)}, target {target} {status} (diff: {diff:+d})")

    print(f"\nExact matches: {total_correct}/{len(image_targets)}")


if __name__ == "__main__":
    test_detection_on_images()
