"""Object-Barcode relationship matching for Task 3."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any

def calculate_spatial_overlap(bbox1: Tuple[int, int, int, int], 
                             bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
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

def calculate_spatial_distance(bbox1: Tuple[int, int, int, int], 
                              bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate center-to-center distance between bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    center1 = (x1 + w1/2, y1 + h1/2)
    center2 = (x2 + w2/2, y2 + h2/2)
    
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance

def match_objects_to_barcodes(object_detections: List[Tuple[int, int, int, int, str, float]],
                             barcode_results: List[Dict],
                             overlap_threshold: float = 0.1,
                             max_distance: float = 200.0) -> Dict[str, Any]:
    """
    Match objects to barcodes based on spatial proximity.
    
    Args:
        object_detections: List of (x, y, w, h, label, confidence) from Task 1
        barcode_results: List of barcode analysis results from Task 2
        overlap_threshold: Minimum IoU for considering a match
        max_distance: Maximum center distance for association
        
    Returns:
        Dictionary with object-barcode associations
    """
    associations = {
        "object_to_barcode": {},  # object_label -> barcode_text
        "barcode_to_object": {},  # barcode_text -> object_label
        "matches": [],            # List of match details
        "unmatched_objects": [],  # Objects without barcodes
        "unmatched_barcodes": []  # Barcodes without objects
    }
    
    # Convert barcode results to bboxes
    barcode_bboxes = []
    for result in barcode_results:
        if result.get('corners') and len(result['corners']) >= 4:
            corners = np.array(result['corners'])
            x_min, y_min = np.min(corners, axis=0)
            x_max, y_max = np.max(corners, axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            barcode_bboxes.append(bbox)
        else:
            barcode_bboxes.append(None)
    
    # Track which items have been matched
    matched_objects = set()
    matched_barcodes = set()
    
    # Find matches based on spatial proximity
    for obj_idx, (obj_x, obj_y, obj_w, obj_h, obj_label, obj_conf) in enumerate(object_detections):
        obj_bbox = (obj_x, obj_y, obj_w, obj_h)
        
        best_barcode_idx = -1
        best_score = 0.0
        best_distance = float('inf')
        
        for bar_idx, result in enumerate(barcode_results):
            if bar_idx in matched_barcodes or barcode_bboxes[bar_idx] is None:
                continue
                
            bar_bbox = barcode_bboxes[bar_idx]
            
            # Calculate overlap and distance
            overlap = calculate_spatial_overlap(obj_bbox, bar_bbox)
            distance = calculate_spatial_distance(obj_bbox, bar_bbox)
            
            # Score based on overlap and proximity
            if overlap >= overlap_threshold or distance <= max_distance:
                # Prefer overlap over proximity
                score = overlap * 2.0 + (1.0 - min(distance / max_distance, 1.0))
                
                if score > best_score:
                    best_score = score
                    best_barcode_idx = bar_idx
                    best_distance = distance
        
        # Create association if match found
        if best_barcode_idx >= 0:
            barcode_result = barcode_results[best_barcode_idx]
            barcode_text = barcode_result.get('decoded_text', 'Unknown')
            
            # Record the match
            match_info = {
                "object_label": obj_label,
                "object_confidence": obj_conf,
                "object_bbox": obj_bbox,
                "barcode_text": barcode_text,
                "barcode_bbox": barcode_bboxes[best_barcode_idx],
                "barcode_confidence": barcode_result.get('confidence_scores', {}).get('overall', 0.0),
                "spatial_overlap": calculate_spatial_overlap(obj_bbox, barcode_bboxes[best_barcode_idx]),
                "spatial_distance": best_distance,
                "match_score": best_score
            }
            
            associations["matches"].append(match_info)
            associations["object_to_barcode"][obj_label] = barcode_text
            associations["barcode_to_object"][barcode_text] = obj_label
            
            matched_objects.add(obj_idx)
            matched_barcodes.add(best_barcode_idx)
    
    # Record unmatched items
    for obj_idx, (_, _, _, _, obj_label, obj_conf) in enumerate(object_detections):
        if obj_idx not in matched_objects:
            associations["unmatched_objects"].append({
                "label": obj_label,
                "confidence": obj_conf,
                "bbox": (object_detections[obj_idx][:4])
            })
    
    for bar_idx, result in enumerate(barcode_results):
        if bar_idx not in matched_barcodes and result.get('decoded_text'):
            associations["unmatched_barcodes"].append({
                "text": result['decoded_text'],
                "confidence": result.get('confidence_scores', {}).get('overall', 0.0),
                "corners": result.get('corners', [])
            })
    
    return associations

def query_relationship(associations: Dict[str, Any], 
                      query: str, 
                      query_type: str = "auto") -> Optional[str]:
    """
    Query object-barcode relationships.
    
    Args:
        associations: Result from match_objects_to_barcodes()
        query: Object name or barcode value to query
        query_type: "object", "barcode", or "auto" for automatic detection
        
    Returns:
        Associated barcode text or object name, or None if not found
    """
    if query_type == "auto":
        # Simple heuristic: if query looks like barcode data, treat as barcode
        if len(query) > 3 and any(c.isdigit() for c in query):
            query_type = "barcode"
        else:
            query_type = "object"
    
    if query_type == "object":
        return associations["object_to_barcode"].get(query)
    elif query_type == "barcode":
        return associations["barcode_to_object"].get(query)
    
    return None

def get_relationship_summary(associations: Dict[str, Any]) -> str:
    """Generate a human-readable summary of object-barcode relationships."""
    summary_lines = []
    
    summary_lines.append(f"Object-Barcode Relationship Summary:")
    summary_lines.append(f"=" * 40)
    
    # Matched pairs
    if associations["matches"]:
        summary_lines.append(f"\nMatched Pairs ({len(associations['matches'])}):")
        for match in associations["matches"]:
            obj_label = match["object_label"]
            barcode_text = match["barcode_text"]
            distance = match["spatial_distance"]
            overlap = match["spatial_overlap"]
            
            summary_lines.append(f"  • {obj_label} ↔ '{barcode_text}'")
            summary_lines.append(f"    Distance: {distance:.1f}px, Overlap: {overlap:.3f}")
    else:
        summary_lines.append(f"\nNo object-barcode matches found.")
    
    # Unmatched items
    if associations["unmatched_objects"]:
        summary_lines.append(f"\nUnmatched Objects ({len(associations['unmatched_objects'])}):")
        for obj in associations["unmatched_objects"]:
            summary_lines.append(f"  • {obj['label']}")
    
    if associations["unmatched_barcodes"]:
        summary_lines.append(f"\nUnmatched Barcodes ({len(associations['unmatched_barcodes'])}):")
        for bar in associations["unmatched_barcodes"]:
            summary_lines.append(f"  • '{bar['text']}'")
    
    return "\n".join(summary_lines)