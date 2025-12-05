"""Complete Task 2 pipeline: Barcode detection, decoding, and 3D normal estimation."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

try:
    from .detection import BarcodeDetector
    from .decoder import Code128Decoder
    from .pose_estimation import BarcodeNormalEstimator
except ImportError:
    # For standalone testing
    from detection import BarcodeDetector
    from decoder import Code128Decoder
    from pose_estimation import BarcodeNormalEstimator

class BarcodeAnalysisPipeline:
    """Complete barcode analysis pipeline for Task 2."""
    
    def __init__(self,
                 detection_params: Optional[Dict] = None,
                 decoding_params: Optional[Dict] = None,
                 pose_params: Optional[Dict] = None):
        """
        Initialize the barcode analysis pipeline.
        
        Args:
            detection_params: Parameters for barcode detection
            decoding_params: Parameters for barcode decoding  
            pose_params: Parameters for pose estimation
        """
        # Default parameters
        detection_params = detection_params or {}
        decoding_params = decoding_params or {}
        pose_params = pose_params or {}
        
        # Initialize components
        self.detector = BarcodeDetector(**detection_params)
        self.decoder = Code128Decoder(**decoding_params)
        self.pose_estimator = BarcodeNormalEstimator(**pose_params)
        
        print("Barcode analysis pipeline initialized!")
    
    def analyze_barcode(self,
                       image: np.ndarray,
                       object_detections: Optional[List[Dict]] = None,
                       object_names: str = "all",
                       debug: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Complete barcode analysis: detection, decoding, and 3D pose estimation.

        Args:
            image: Input image (BGR format)
            object_detections: Optional list of object detections from Task 1.
                Each detection should have 'name' and 'bbox' (x, y, w, h).
                If provided, barcode detection is constrained to these regions.
            object_names: "all" or list of object names to process.
                Only used when object_detections is provided.
            debug: Whether to include debug information

        Returns:
            Tuple of (barcode_results, pipeline_debug_info)
            - barcode_results: List of analysis results for each detected barcode
            - pipeline_debug_info: Overall pipeline execution information
        """
        pipeline_debug = {
            "input_shape": image.shape,
            "timings": {},
            "detection_count": 0,
            "successful_decodings": 0,
            "successful_pose_estimations": 0,
            "object_constrained": object_detections is not None
        }

        # Step 1: Detect barcode regions
        print("Detecting barcode regions...")
        start_time = time.time()

        if object_detections is not None:
            # Object-constrained detection (integrate with Task 1)
            print(f"  Using {len(object_detections)} object regions from Task 1")
            barcode_detections = self.detector.detect_in_object_regions(
                image, object_detections, object_names=object_names, one_barcode_per_object=False
            )
            detection_debug = {"mode": "object_constrained", "object_count": len(object_detections)}
        else:
            # Full image detection
            barcode_detections, detection_debug = self.detector.detect_barcodes(image, debug=debug)

        pipeline_debug["timings"]["detection"] = time.time() - start_time
        pipeline_debug["detection_count"] = len(barcode_detections)

        if debug:
            pipeline_debug["detection_debug"] = detection_debug

        print(f"Found {len(barcode_detections)} potential barcode regions")

        if not barcode_detections:
            print("No barcode regions detected!")
            return [], pipeline_debug

        # Convert detection format if needed (new format returns dicts, old format returns arrays)
        barcode_regions = []
        for det in barcode_detections:
            if isinstance(det, dict):
                barcode_regions.append(det)
            else:
                # Legacy format - convert to dict
                barcode_regions.append({'box': det, 'center': tuple(np.mean(det, axis=0))})
        
        # Step 2: Process each detected region
        barcode_results = []

        for i, detection in enumerate(barcode_regions):
            print(f"Processing barcode region {i+1}/{len(barcode_regions)}...")

            # Extract corners from detection dict or use directly if array
            if isinstance(detection, dict):
                region_corners = np.array(detection['box'])
                object_name = detection.get('object_name', None)
            else:
                region_corners = detection
                object_name = None

            region_result = {
                "id": i,
                "corners": region_corners.tolist(),
                "object_name": object_name,
                "decoded_text": None,
                "normal_vector": None,
                "confidence_scores": {},
                "debug_info": {}
            }

            try:
                # Extract and rectify barcode region
                start_time = time.time()
                rectified_barcode = self.detector.extract_barcode_region(
                    image, detection, output_size=(300, 100))
                
                region_result["debug_info"]["rectification_time"] = time.time() - start_time
                
                if debug:
                    region_result["debug_info"]["rectified_image"] = rectified_barcode
                
                # Step 3: Decode the barcode
                start_time = time.time()
                decoded_text, decoding_debug = self.decoder.decode_barcode(
                    rectified_barcode, debug=debug)
                
                region_result["debug_info"]["decoding_time"] = time.time() - start_time
                region_result["decoded_text"] = decoded_text
                
                if debug:
                    region_result["debug_info"]["decoding"] = decoding_debug
                
                if decoded_text:
                    pipeline_debug["successful_decodings"] += 1
                    print(f"  Decoded: '{decoded_text}'")
                else:
                    print(f"  Decoding failed: {decoding_debug.get('error', 'Unknown error')}")
                
                # Step 4: Estimate 3D pose and normal vector
                start_time = time.time()
                normal_vector, pose_debug = self.pose_estimator.estimate_normal_vector(
                    image, region_corners)
                
                region_result["debug_info"]["pose_estimation_time"] = time.time() - start_time
                region_result["normal_vector"] = normal_vector.tolist() if normal_vector is not None else None
                
                if debug:
                    region_result["debug_info"]["pose_estimation"] = pose_debug
                
                if normal_vector is not None:
                    pipeline_debug["successful_pose_estimations"] += 1
                    print(f"  Normal vector: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")
                    
                    # Calculate confidence scores
                    region_result["confidence_scores"] = self._calculate_confidence_scores(
                        decoding_debug, pose_debug)
                else:
                    print(f"  Pose estimation failed: {pose_debug.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"  Error processing region {i}: {e}")
                region_result["debug_info"]["error"] = str(e)
            
            barcode_results.append(region_result)
        
        # Calculate total timing
        pipeline_debug["timings"]["total"] = sum(pipeline_debug["timings"].values())
        
        print(f"\nPipeline completed:")
        print(f"  Detected: {pipeline_debug['detection_count']} regions")
        print(f"  Decoded: {pipeline_debug['successful_decodings']} barcodes")
        print(f"  Pose estimated: {pipeline_debug['successful_pose_estimations']} regions")
        print(f"  Total time: {pipeline_debug['timings']['total']:.3f}s")
        
        return barcode_results, pipeline_debug
    
    def visualize_results(self, 
                         image: np.ndarray,
                         barcode_results: List[Dict],
                         show_details: bool = True) -> np.ndarray:
        """
        Visualize barcode analysis results on the image.
        
        Args:
            image: Original input image
            barcode_results: Results from analyze_barcode()
            show_details: Whether to show detailed annotations
            
        Returns:
            Image with visualization overlays
        """
        result_img = image.copy()
        
        for i, result in enumerate(barcode_results):
            corners = np.array(result["corners"], dtype=np.int32)
            decoded_text = result["decoded_text"]
            normal_vector = result["normal_vector"]
            
            # Draw barcode region outline
            color = (0, 255, 0) if decoded_text else (0, 0, 255)  # Green if decoded, red if not
            cv2.drawContours(result_img, [corners], -1, color, 2)
            
            # Add region ID
            center = np.mean(corners, axis=0).astype(int)
            cv2.putText(result_img, f"#{i+1}", (center[0], center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if show_details:
                # Add decoded text
                if decoded_text:
                    text_pos = (corners[0][0], corners[0][1] - 10)
                    cv2.putText(result_img, f"Text: {decoded_text}", text_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw normal vector arrow
                if normal_vector:
                    normal_np = np.array(normal_vector)
                    result_img = self.pose_estimator.visualize_normal_vector(
                        result_img, corners, normal_np, scale=50)
        
        return result_img
    
    def _calculate_confidence_scores(self, 
                                   decoding_debug: Dict,
                                   pose_debug: Dict) -> Dict:
        """Calculate confidence scores for the analysis results."""
        scores = {
            "overall": 0.0,
            "decoding": 0.0,
            "pose_estimation": 0.0
        }
        
        # Decoding confidence
        if decoding_debug.get("checksum_valid", False):
            scores["decoding"] = 0.9  # High confidence for valid checksum
        elif len(decoding_debug.get("detected_characters", [])) > 0:
            scores["decoding"] = 0.5  # Medium confidence for partial decode
        else:
            scores["decoding"] = 0.0  # No confidence for failed decode
        
        # Pose estimation confidence
        if pose_debug.get("success", False):
            reprojection_error = pose_debug.get("reprojection_error", float('inf'))
            if reprojection_error < 5.0:
                scores["pose_estimation"] = 0.9
            elif reprojection_error < 15.0:
                scores["pose_estimation"] = 0.7
            elif reprojection_error < 30.0:
                scores["pose_estimation"] = 0.5
            else:
                scores["pose_estimation"] = 0.3
        else:
            scores["pose_estimation"] = 0.0
        
        # Overall confidence (weighted average)
        scores["overall"] = (scores["decoding"] * 0.6 + scores["pose_estimation"] * 0.4)
        
        return scores
    
    def get_best_barcode(self, barcode_results: List[Dict]) -> Optional[Dict]:
        """Get the barcode result with highest confidence score."""
        if not barcode_results:
            return None
        
        best_result = max(barcode_results, 
                         key=lambda x: x.get("confidence_scores", {}).get("overall", 0.0))
        
        return best_result if best_result.get("confidence_scores", {}).get("overall", 0.0) > 0.1 else None
    
    def export_results(self, 
                      barcode_results: List[Dict],
                      pipeline_debug: Dict,
                      output_path: str) -> None:
        """Export analysis results to JSON file."""
        import json
        
        export_data = {
            "pipeline_info": {
                "detection_count": pipeline_debug["detection_count"],
                "successful_decodings": pipeline_debug["successful_decodings"],
                "successful_pose_estimations": pipeline_debug["successful_pose_estimations"],
                "total_time": pipeline_debug["timings"]["total"]
            },
            "barcode_results": []
        }
        
        for result in barcode_results:
            # Clean result for JSON export (remove debug images)
            clean_result = {
                "id": result["id"],
                "corners": result["corners"],
                "decoded_text": result["decoded_text"],
                "normal_vector": result["normal_vector"],
                "confidence_scores": result["confidence_scores"]
            }
            export_data["barcode_results"].append(clean_result)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to: {output_path}")

def demo_task2_pipeline(image_path: str) -> None:
    """Demonstrate complete Task 2 pipeline on a test image."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Create pipeline
    pipeline = BarcodeAnalysisPipeline(
        detection_params={"min_area": 500, "max_area": 50000},
        pose_params={"barcode_real_size": (40.0, 12.0)}
    )
    
    # Run complete analysis
    results, debug_info = pipeline.analyze_barcode(image, debug=True)
    
    # Visualize results
    result_image = pipeline.visualize_results(image, results)
    
    # Save visualization
    output_path = f"demo_task2_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Visualization saved: {output_path}")
    
    # Export results
    pipeline.export_results(results, debug_info, "demo_task2_results.json")
    
    # Show best result
    best_result = pipeline.get_best_barcode(results)
    if best_result:
        print(f"\nBest barcode result:")
        print(f"  Text: {best_result['decoded_text']}")
        print(f"  Normal: {best_result['normal_vector']}")
        print(f"  Confidence: {best_result['confidence_scores']['overall']:.3f}")

if __name__ == "__main__":
    # Demo the complete Task 2 pipeline
    test_images = [
        "../../Images/input_1.jpg",
        "../../Images/input_2.jpg", 
        "../../Images/input_3.jpg"
    ]
    
    for img_path in test_images:
        print(f"\n{'='*50}")
        print(f"Testing Task 2 pipeline with: {img_path}")
        print('='*50)
        
        try:
            demo_task2_pipeline(img_path)
        except Exception as e:
            print(f"Error testing {img_path}: {e}")
        
        break  # Test just the first image for demo