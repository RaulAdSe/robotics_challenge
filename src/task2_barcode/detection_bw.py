"""Black-on-white barcode detection optimized for Code128 barcodes.

This module detects Code128 barcodes by specifically looking for:
- High contrast regions with vertical OR horizontal bar patterns
- Black parallel lines on white/light background
- Regular transition patterns typical of barcodes
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math


class BlackWhiteBarcodeDetector:
    """Detect black-on-white barcodes using gradient and variance analysis.

    This detector handles both horizontal barcodes (vertical bars) and
    vertical barcodes (horizontal bars).
    """

    def __init__(self,
                 min_aspect: float = 1.3,
                 max_aspect: float = 8.0,
                 min_area_pct: float = 0.0003,
                 max_area_pct: float = 0.5,
                 min_solidity: float = 0.5,
                 min_transitions: int = 8,   # Optimal from tuning
                 min_contrast: int = 30,     # Optimal from tuning
                 min_score: float = 0.4):    # Optimal from tuning
        """
        Initialize black-on-white barcode detector.

        Args:
            min_aspect: Minimum aspect ratio (longer/shorter side)
            max_aspect: Maximum aspect ratio
            min_area_pct: Minimum area as percentage of image
            max_area_pct: Maximum area as percentage of image
            min_solidity: Minimum solidity for rectangular shape
            min_transitions: Minimum black/white transitions for barcode
            min_contrast: Minimum contrast in barcode region
            min_score: Minimum score to accept as barcode
        """
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct
        self.min_solidity = min_solidity
        self.min_transitions = min_transitions
        self.min_contrast = min_contrast
        self.min_score = min_score

    def detect_barcodes(self, image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None,
                        debug: bool = False,
                        use_absolute_area: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Detect barcode regions in an image.

        Args:
            image: Input image (BGR format)
            roi: Optional region of interest as (x, y, width, height)
            debug: Whether to return debug information
            use_absolute_area: If True and roi specified, use full image area

        Returns:
            Tuple of (barcode_detections, debug_info)
        """
        debug_info = {
            "input_shape": image.shape,
            "roi": roi,
            "processing_stages": {},
            "detected_count": 0
        }

        full_img_area = image.shape[0] * image.shape[1]

        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi
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
        img_area = full_img_area if (roi is not None and use_absolute_area) else img_h * img_w

        # Convert to grayscale and enhance
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        if debug:
            debug_info["processing_stages"]["enhanced"] = enhanced.copy()

        # Compute gradients
        grad_x = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
        grad_x_abs = np.abs(grad_x)
        grad_y_abs = np.abs(grad_y)

        # Local variance for contrast detection
        ksize = 15
        mean = cv2.blur(enhanced.astype(np.float32), (ksize, ksize))
        sq_mean = cv2.blur(enhanced.astype(np.float32)**2, (ksize, ksize))
        variance = np.clip(sq_mean - mean**2, 0, None)
        var_norm = variance / (np.max(variance) + 1e-6) * 255

        candidates = []

        # Process both gradient directions to handle both barcode orientations
        for grad_name, grad_abs, kernel_close in [
            ('horizontal', grad_x_abs, (15, 1)),   # Detects vertical bars
            ('vertical', grad_y_abs, (1, 15))      # Detects horizontal bars
        ]:
            grad_norm = grad_abs / (np.max(grad_abs) + 1e-6) * 255
            combined = np.minimum(grad_norm, var_norm).astype(np.uint8)

            _, binary = cv2.threshold(combined, 15, 255, cv2.THRESH_BINARY)  # Lowered from 25

            # Morphological operations
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_close)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_close[::-1])
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)

            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
            dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_open)

            if debug:
                debug_info["processing_stages"][f"mask_{grad_name}"] = opened.copy()

            # Find contours
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = int(img_area * self.min_area_pct)
            max_area = int(img_area * self.max_area_pct)

            for cnt in contours:
                candidate = self._validate_contour(
                    enhanced, cnt, min_area, max_area, roi_offset
                )
                if candidate is not None:
                    candidates.append(candidate)

        # Non-maximum suppression
        detections = self._nms(candidates)

        debug_info["detected_count"] = len(detections)
        debug_info["candidates_before_nms"] = len(candidates)

        return detections, debug_info

    def _validate_contour(self, gray: np.ndarray, contour: np.ndarray,
                          min_area: int, max_area: int,
                          roi_offset: Tuple[int, int]) -> Optional[Dict]:
        """Validate a contour as a potential barcode."""
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            return None

        x, y, bw, bh = cv2.boundingRect(contour)

        if bw < 20 or bh < 20:  # Lowered from 30
            return None

        aspect = max(bw, bh) / min(bw, bh)

        if aspect < self.min_aspect or aspect > self.max_aspect:
            return None

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity < self.min_solidity:
            return None

        # Extract and validate pattern
        region = gray[y:y+bh, x:x+bw]
        if region.size == 0:
            return None

        is_vertical_barcode = bh > bw
        score, info = self._verify_pattern(region, is_vertical_barcode)

        if score < self.min_score:
            return None

        # Get rotated rect for better box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Adjust for ROI offset
        box = box + np.array([roi_offset[0], roi_offset[1]])
        center = (x + bw//2 + roi_offset[0], y + bh//2 + roi_offset[1])

        return {
            'box': box,
            'bbox': (x + roi_offset[0], y + roi_offset[1], bw, bh),
            'center': center,
            'size': (max(bw, bh), min(bw, bh)),
            'angle': rect[2],
            'area': area,
            'aspect': aspect,
            'solidity': solidity,
            'extent': area / (bw * bh) if bw * bh > 0 else 0,
            'orientation': 'vertical' if is_vertical_barcode else 'horizontal',
            'contour': contour,
            'detection_method': 'gradient_variance',
            'barcode_score': score,
            'transitions': info.get('transitions', 0),
            'contrast': info.get('contrast', 0)
        }

    def _verify_pattern(self, region: np.ndarray,
                        is_vertical: bool) -> Tuple[float, Dict]:
        """Verify barcode pattern with strict criteria."""
        h, w = region.shape[:2]

        if is_vertical:
            if h < 25 or w < 15:  # Lowered
                return 0.0, {}
            profiles = []
            for off in [-w//4, 0, w//4]:
                col = max(0, min(w-1, w//2 + off))
                profiles.append(region[:, col].astype(np.float32))
            profile = np.mean(profiles, axis=0)
        else:
            if w < 25 or h < 15:  # Lowered
                return 0.0, {}
            profiles = []
            for off in [-h//4, 0, h//4]:
                row = max(0, min(h-1, h//2 + off))
                profiles.append(region[row, :].astype(np.float32))
            profile = np.mean(profiles, axis=0)

        min_val, max_val = np.min(profile), np.max(profile)
        contrast = int(max_val - min_val)

        if contrast < self.min_contrast:
            return 0.0, {'contrast': contrast}

        thresh = (min_val + max_val) / 2
        binary = (profile < thresh).astype(np.uint8)

        transitions = int(np.sum(np.abs(np.diff(binary))))

        if transitions < self.min_transitions:
            return 0.0, {'transitions': transitions}

        # Check regularity
        diff = np.diff(binary)
        positions = np.where(np.abs(diff) > 0)[0]

        if len(positions) < 8:
            return 0.0, {'bars': len(positions)}

        gaps = np.diff(positions)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        cv = std_gap / (mean_gap + 1e-6)

        if cv > 1.5:
            return 0.3, {'cv': cv, 'transitions': transitions}

        white_ratio = np.sum(binary == 0) / len(binary)
        if white_ratio < 0.25 or white_ratio > 0.75:
            return 0.3, {'white_ratio': white_ratio}

        # Calculate score
        score = 0.6

        if transitions >= 30:
            score += 0.2
        elif transitions >= 20:
            score += 0.1

        if contrast >= 100:
            score += 0.1

        if cv < 0.8:
            score += 0.1

        return min(1.0, score), {
            'transitions': transitions,
            'contrast': contrast,
            'cv': round(cv, 2),
            'white_ratio': round(white_ratio, 2)
        }

    def _nms(self, candidates: List[Dict],
             distance_ratio: float = 0.7) -> List[Dict]:
        """Non-maximum suppression."""
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda x: x['barcode_score'], reverse=True)

        kept = []
        for cand in candidates:
            is_duplicate = False
            for kept_cand in kept:
                dx = cand['center'][0] - kept_cand['center'][0]
                dy = cand['center'][1] - kept_cand['center'][1]
                dist = np.sqrt(dx*dx + dy*dy)

                min_dim = min(cand['size'][1], kept_cand['size'][1])
                if dist < min_dim * distance_ratio:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(cand)

        return kept

    def detect_in_object_regions(self, image: np.ndarray,
                                  object_detections: List[Dict],
                                  object_names: Union[str, List[str]] = "all",
                                  expand_ratio: float = 0.15,
                                  one_barcode_per_object: bool = True) -> List[Dict]:
        """Detect barcodes within detected object regions from Task 1."""
        all_detections = []

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

            expand_x = int(w * expand_ratio)
            expand_y = int(h * expand_ratio)

            expanded_roi = (
                max(0, x - expand_x),
                max(0, y - expand_y),
                min(w + 2 * expand_x, image.shape[1] - max(0, x - expand_x)),
                min(h + 2 * expand_y, image.shape[0] - max(0, y - expand_y))
            )

            detections, _ = self.detect_barcodes(
                image, roi=expanded_roi, use_absolute_area=True
            )

            if not detections:
                continue

            if one_barcode_per_object:
                best_det = max(detections,
                              key=lambda d: d['barcode_score'] * d['solidity'])
                best_det['object_name'] = obj_name
                best_det['object_bbox'] = bbox
                all_detections.append(best_det)
            else:
                for det in detections:
                    det['object_name'] = obj_name
                    det['object_bbox'] = bbox
                    all_detections.append(det)

        return self._nms(all_detections)

    def extract_barcode_region(self, image: np.ndarray,
                               detection: Dict,
                               output_size: Tuple[int, int] = (300, 100)) -> np.ndarray:
        """Extract and deskew barcode region from image."""
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
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        center = np.mean(corners, axis=0)

        angles = []
        for corner in corners:
            angle = math.atan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)

        sorted_indices = np.argsort(angles)
        ordered = corners[sorted_indices]

        return ordered


# Keep the original BarcodeDetector class name for backward compatibility
BarcodeDetector = BlackWhiteBarcodeDetector


def test_bw_detection():
    """Test black-on-white barcode detection."""
    import os

    detector = BlackWhiteBarcodeDetector()

    image_files = [f'Images/input_{i}.jpg' for i in range(1, 9)]

    print("Testing black-on-white barcode detection:")
    print("=" * 70)

    total_detected = 0

    for img_path in image_files:
        if not os.path.exists(img_path):
            print(f"  {img_path}: FILE NOT FOUND")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  {img_path}: COULD NOT LOAD")
            continue

        detections, debug_info = detector.detect_barcodes(image, debug=True)

        total_detected += len(detections)

        print(f"\n{img_path}: detected {len(detections)} barcodes")

        for i, det in enumerate(detections):
            x, y, bw, bh = det['bbox']
            print(f"  #{i+1} [{det['orientation']}]: ({x},{y},{bw}x{bh}), "
                  f"score={det['barcode_score']:.2f}, "
                  f"transitions={det['transitions']}, contrast={det['contrast']}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: Detected {total_detected} barcodes across all images")


if __name__ == "__main__":
    test_bw_detection()
