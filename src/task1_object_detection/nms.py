"""Non-Maximum Suppression implementation for object detection."""

import numpy as np
from typing import List, Tuple

def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes as (x, y, width, height)
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Check if there's any intersection
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    # Calculate intersection area
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0.0

def non_max_suppression(detections: List[Tuple[int, int, int, int, str, float]], 
                       iou_threshold: float = 0.5,
                       score_threshold: float = 0.0) -> List[Tuple[int, int, int, int, str, float]]:
    """
    Apply Non-Maximum Suppression to detection results.
    
    Args:
        detections: List of (x, y, w, h, label, confidence) tuples
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum confidence score to keep detection
        
    Returns:
        Filtered list of detections after NMS
    """
    if not detections:
        return []
    
    # Filter by confidence threshold first
    filtered_detections = [det for det in detections if det[5] >= score_threshold]
    
    if not filtered_detections:
        return []
    
    # Group detections by class label
    class_groups = {}
    for detection in filtered_detections:
        label = detection[4]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(detection)
    
    # Apply NMS to each class separately
    final_detections = []
    for label, class_detections in class_groups.items():
        class_nms_results = _nms_single_class(class_detections, iou_threshold)
        final_detections.extend(class_nms_results)
    
    # Sort final results by confidence (highest first)
    final_detections.sort(key=lambda x: x[5], reverse=True)
    
    return final_detections

def _nms_single_class(detections: List[Tuple[int, int, int, int, str, float]], 
                     iou_threshold: float) -> List[Tuple[int, int, int, int, str, float]]:
    """
    Apply NMS to detections of a single class.
    
    Args:
        detections: List of detections for a single class
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered detections after NMS
    """
    if not detections:
        return []
    
    # Sort by confidence score (highest first)
    sorted_detections = sorted(detections, key=lambda x: x[5], reverse=True)
    
    kept_detections = []
    suppressed = [False] * len(sorted_detections)
    
    for i, current_detection in enumerate(sorted_detections):
        if suppressed[i]:
            continue
        
        # Keep current detection
        kept_detections.append(current_detection)
        current_box = current_detection[:4]
        
        # Suppress overlapping detections
        for j in range(i + 1, len(sorted_detections)):
            if suppressed[j]:
                continue
            
            other_box = sorted_detections[j][:4]
            iou = calculate_iou(current_box, other_box)
            
            if iou > iou_threshold:
                suppressed[j] = True
    
    return kept_detections

def soft_nms(detections: List[Tuple[int, int, int, int, str, float]], 
            sigma: float = 0.5,
            iou_threshold: float = 0.3,
            score_threshold: float = 0.001) -> List[Tuple[int, int, int, int, str, float]]:
    """
    Apply Soft Non-Maximum Suppression to detection results.
    
    Instead of completely removing overlapping boxes, Soft NMS reduces their
    confidence scores based on overlap amount.
    
    Args:
        detections: List of (x, y, w, h, label, confidence) tuples
        sigma: Gaussian parameter for score decay
        iou_threshold: IoU threshold for applying decay
        score_threshold: Final minimum score to keep detection
        
    Returns:
        Filtered list of detections after Soft NMS
    """
    if not detections:
        return []
    
    # Group by class
    class_groups = {}
    for detection in detections:
        label = detection[4]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(detection)
    
    # Apply Soft NMS to each class
    final_detections = []
    for label, class_detections in class_groups.items():
        class_soft_nms_results = _soft_nms_single_class(
            class_detections, sigma, iou_threshold, score_threshold)
        final_detections.extend(class_soft_nms_results)
    
    # Sort by confidence
    final_detections.sort(key=lambda x: x[5], reverse=True)
    
    return final_detections

def _soft_nms_single_class(detections: List[Tuple[int, int, int, int, str, float]], 
                          sigma: float,
                          iou_threshold: float,
                          score_threshold: float) -> List[Tuple[int, int, int, int, str, float]]:
    """Apply Soft NMS to a single class."""
    if not detections:
        return []
    
    # Convert to mutable list for score updates
    working_detections = [list(det) for det in detections]
    
    # Sort by confidence
    working_detections.sort(key=lambda x: x[5], reverse=True)
    
    kept_detections = []
    
    while working_detections:
        # Take highest confidence detection
        current = working_detections.pop(0)
        
        # Only keep if above threshold
        if current[5] >= score_threshold:
            kept_detections.append(tuple(current))
        
        # Update scores of remaining detections
        current_box = current[:4]
        
        for other in working_detections:
            other_box = other[:4]
            iou = calculate_iou(current_box, other_box)
            
            if iou > iou_threshold:
                # Apply Gaussian decay
                other[5] *= np.exp(-(iou * iou) / sigma)
    
    return kept_detections

def distance_based_nms(detections: List[Tuple[int, int, int, int, str, float]], 
                      distance_threshold: float = 50.0,
                      score_threshold: float = 0.0) -> List[Tuple[int, int, int, int, str, float]]:
    """
    Apply distance-based NMS using center points instead of IoU.
    
    Useful when objects might not have significant overlap but are very close.
    
    Args:
        detections: List of (x, y, w, h, label, confidence) tuples
        distance_threshold: Maximum center distance to consider for suppression
        score_threshold: Minimum confidence score
        
    Returns:
        Filtered detections
    """
    if not detections:
        return []
    
    # Filter by score first
    filtered = [det for det in detections if det[5] >= score_threshold]
    
    # Group by class
    class_groups = {}
    for detection in filtered:
        label = detection[4]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(detection)
    
    # Apply distance NMS to each class
    final_detections = []
    for label, class_detections in class_groups.items():
        class_results = _distance_nms_single_class(class_detections, distance_threshold)
        final_detections.extend(class_results)
    
    final_detections.sort(key=lambda x: x[5], reverse=True)
    return final_detections

def _distance_nms_single_class(detections: List[Tuple[int, int, int, int, str, float]], 
                             distance_threshold: float) -> List[Tuple[int, int, int, int, str, float]]:
    """Apply distance-based NMS to single class."""
    if not detections:
        return []
    
    # Sort by confidence
    sorted_detections = sorted(detections, key=lambda x: x[5], reverse=True)
    
    kept_detections = []
    
    for current in sorted_detections:
        x, y, w, h, label, conf = current
        current_center = (x + w/2, y + h/2)
        
        # Check if too close to any kept detection
        should_keep = True
        for kept in kept_detections:
            kx, ky, kw, kh, klabel, kconf = kept
            if klabel != label:  # Only compare same class
                continue
                
            kept_center = (kx + kw/2, ky + kh/2)
            distance = np.sqrt((current_center[0] - kept_center[0])**2 + 
                             (current_center[1] - kept_center[1])**2)
            
            if distance < distance_threshold:
                should_keep = False
                break
        
        if should_keep:
            kept_detections.append(current)
    
    return kept_detections

def analyze_detection_overlap(detections: List[Tuple[int, int, int, int, str, float]]) -> dict:
    """
    Analyze overlap patterns in detections for debugging NMS parameters.
    
    Args:
        detections: List of detections
        
    Returns:
        Dictionary with overlap statistics
    """
    if len(detections) < 2:
        return {"max_iou": 0.0, "mean_iou": 0.0, "overlap_pairs": 0}
    
    ious = []
    overlap_pairs = 0
    
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            box1 = detections[i][:4]
            box2 = detections[j][:4]
            iou = calculate_iou(box1, box2)
            ious.append(iou)
            
            if iou > 0.1:  # Significant overlap
                overlap_pairs += 1
    
    return {
        "max_iou": max(ious) if ious else 0.0,
        "mean_iou": np.mean(ious) if ious else 0.0,
        "overlap_pairs": overlap_pairs,
        "total_pairs": len(ious)
    }

if __name__ == "__main__":
    # Test NMS functions
    test_detections = [
        (100, 100, 50, 50, "car", 0.9),
        (105, 105, 45, 45, "car", 0.8),  # Overlapping car
        (200, 200, 60, 60, "person", 0.7),
        (300, 300, 40, 40, "car", 0.6),
        (205, 205, 55, 55, "person", 0.65),  # Overlapping person
    ]
    
    print("Original detections:", len(test_detections))
    
    # Test standard NMS
    nms_result = non_max_suppression(test_detections, iou_threshold=0.5)
    print("After NMS:", len(nms_result))
    
    # Test Soft NMS
    soft_nms_result = soft_nms(test_detections)
    print("After Soft NMS:", len(soft_nms_result))
    
    # Analyze overlaps
    overlap_stats = analyze_detection_overlap(test_detections)
    print("Overlap analysis:", overlap_stats)