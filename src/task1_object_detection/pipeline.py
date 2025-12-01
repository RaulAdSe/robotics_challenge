"""Complete Task 1 pipeline: Object detection with text input using region proposals + CLIP."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time

from .region_proposals import RegionProposalGenerator
from .clip_classifier import CLIPClassifier
from .nms import non_max_suppression, soft_nms

class ObjectDetectionPipeline:
    """Complete object detection pipeline for Task 1."""
    
    def __init__(self, 
                 proposal_method: str = "selective_search",
                 clip_model: str = "ViT-B/32",
                 confidence_threshold: float = 0.1,
                 nms_threshold: float = 0.5,
                 max_proposals: int = 1000,
                 min_proposal_size: int = 500,
                 use_soft_nms: bool = False):
        """
        Initialize the object detection pipeline.
        
        Args:
            proposal_method: "selective_search", "blob_detection", or "both"
            clip_model: CLIP model variant to use
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
            max_proposals: Maximum number of region proposals
            min_proposal_size: Minimum area for proposals
            use_soft_nms: Whether to use Soft NMS instead of standard NMS
        """
        self.proposal_method = proposal_method
        self.clip_model = clip_model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_proposals = max_proposals
        self.min_proposal_size = min_proposal_size
        self.use_soft_nms = use_soft_nms
        
        # Initialize components
        print("Initializing object detection pipeline...")
        
        self.proposal_generator = RegionProposalGenerator(
            method=proposal_method,
            min_size=min_proposal_size,
            max_proposals=max_proposals
        )
        
        # CLIP classifier will be initialized on first use to avoid GPU memory
        # allocation if not needed
        self.clip_classifier = None
        
        print("Pipeline initialization complete!")
    
    def _ensure_clip_loaded(self):
        """Ensure CLIP classifier is loaded (lazy loading)."""
        if self.clip_classifier is None:
            self.clip_classifier = CLIPClassifier(
                model_name=self.clip_model,
                confidence_threshold=self.confidence_threshold
            )
    
    def detect_objects(self, 
                      image: np.ndarray, 
                      object_names: List[str],
                      debug: bool = False) -> Tuple[List[Tuple[int, int, int, int, str, float]], Dict]:
        """
        Detect objects in image based on text descriptions.
        
        Args:
            image: Input image (BGR format)
            object_names: List of object names to search for
            debug: Whether to return debug information
            
        Returns:
            Tuple of (detections, debug_info)
            - detections: List of (x, y, w, h, label, confidence)
            - debug_info: Dictionary with timing and intermediate results
        """
        debug_info = {
            "input_shape": image.shape,
            "object_names": object_names,
            "timings": {},
            "proposal_count": 0,
            "classification_count": 0,
            "final_count": 0
        }
        
        # Step 1: Generate region proposals
        start_time = time.time()
        print("Generating region proposals...")
        
        proposals = self.proposal_generator.generate_proposals(image)
        debug_info["proposal_count"] = len(proposals)
        debug_info["timings"]["proposal_generation"] = time.time() - start_time
        
        print(f"Generated {len(proposals)} region proposals")
        
        if not proposals:
            print("No valid proposals found!")
            return [], debug_info
        
        # Step 2: Classify regions using CLIP
        start_time = time.time()
        print("Classifying regions with CLIP...")
        
        self._ensure_clip_loaded()
        
        # Use batch processing for better efficiency
        classifications = self.clip_classifier.classify_regions_batch(
            image, proposals, object_names, batch_size=32
        )
        
        debug_info["classification_count"] = len(classifications)
        debug_info["timings"]["classification"] = time.time() - start_time
        
        print(f"Found {len(classifications)} confident classifications")
        
        if not classifications:
            print("No confident classifications found!")
            return [], debug_info
        
        # Step 3: Apply Non-Maximum Suppression
        start_time = time.time()
        print("Applying Non-Maximum Suppression...")
        
        if self.use_soft_nms:
            final_detections = soft_nms(
                classifications,
                sigma=0.5,
                iou_threshold=self.nms_threshold,
                score_threshold=self.confidence_threshold
            )
        else:
            final_detections = non_max_suppression(
                classifications,
                iou_threshold=self.nms_threshold,
                score_threshold=self.confidence_threshold
            )
        
        debug_info["final_count"] = len(final_detections)
        debug_info["timings"]["nms"] = time.time() - start_time
        debug_info["timings"]["total"] = sum(debug_info["timings"].values())
        
        print(f"Final detections after NMS: {len(final_detections)}")
        
        # Store intermediate results for debugging
        if debug:
            debug_info["proposals"] = proposals[:50]  # Store first 50 for analysis
            debug_info["raw_classifications"] = classifications
        
        return final_detections, debug_info
    
    def detect_objects_simple(self, 
                            image: np.ndarray, 
                            object_names: List[str]) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Simplified detection interface without debug info.
        
        Args:
            image: Input image (BGR format)
            object_names: List of object names to search for
            
        Returns:
            List of (x, y, w, h, label, confidence) detections
        """
        detections, _ = self.detect_objects(image, object_names, debug=False)
        return detections
    
    def benchmark_pipeline(self, 
                         image: np.ndarray, 
                         object_names: List[str],
                         num_runs: int = 3) -> Dict:
        """
        Benchmark pipeline performance.
        
        Args:
            image: Test image
            object_names: Objects to detect
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark statistics
        """
        times = []
        detection_counts = []
        
        for i in range(num_runs):
            print(f"Benchmark run {i+1}/{num_runs}...")
            start_time = time.time()
            
            detections, debug_info = self.detect_objects(image, object_names)
            
            total_time = time.time() - start_time
            times.append(total_time)
            detection_counts.append(len(detections))
        
        return {
            "runs": num_runs,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "mean_detections": np.mean(detection_counts),
            "detection_consistency": np.std(detection_counts)
        }
    
    def get_pipeline_info(self) -> Dict:
        """Get information about pipeline configuration."""
        info = {
            "proposal_method": self.proposal_method,
            "clip_model": self.clip_model,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "max_proposals": self.max_proposals,
            "min_proposal_size": self.min_proposal_size,
            "use_soft_nms": self.use_soft_nms
        }
        
        if self.clip_classifier is not None:
            info.update(self.clip_classifier.get_model_info())
        
        return info

def demo_pipeline(image_path: str, object_names: List[str]) -> None:
    """
    Demonstrate the object detection pipeline on a test image.
    
    Args:
        image_path: Path to test image
        object_names: List of objects to detect
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Loaded image: {image.shape}")
    print(f"Looking for objects: {object_names}")
    
    # Create pipeline
    pipeline = ObjectDetectionPipeline(
        proposal_method="blob_detection",  # Use blob detection for demo (faster)
        confidence_threshold=0.05,
        use_soft_nms=False
    )
    
    # Run detection
    detections, debug_info = pipeline.detect_objects(image, object_names, debug=True)
    
    # Print results
    print(f"\nDetection Results:")
    print(f"Timing breakdown:")
    for stage, time_taken in debug_info["timings"].items():
        print(f"  {stage}: {time_taken:.3f}s")
    
    print(f"\nFound {len(detections)} objects:")
    for i, (x, y, w, h, label, conf) in enumerate(detections):
        print(f"  {i+1}. {label}: ({x},{y}) {w}x{h} confidence={conf:.3f}")
    
    # Visualize results
    from ..utils.visualization import draw_bounding_boxes
    
    result_image = draw_bounding_boxes(
        image, 
        [(x, y, w, h) for x, y, w, h, _, _ in detections],
        [f"{label} ({conf:.2f})" for _, _, _, _, label, conf in detections]
    )
    
    output_path = f"demo_detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nVisualization saved: {output_path}")

if __name__ == "__main__":
    # Demo the pipeline with a test image
    demo_pipeline(
        "Images/input_1.jpg",
        ["person", "car", "bicycle", "dog", "bottle", "chair", "laptop"]
    )