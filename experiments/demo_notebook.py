"""
Demo notebook for testing both Task 1 and Task 2 pipelines.

This can be converted to a Jupyter notebook for interactive exploration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import our pipelines
from task1_object_detection.pipeline import ObjectDetectionPipeline
from task2_barcode.pipeline import BarcodeAnalysisPipeline

def demo_task1():
    """Demonstrate Task 1 - Object Detection."""
    print("=== TASK 1 DEMO: Object Detection ===")
    
    # Test image path
    test_image_path = "../Images/input_1.jpg"
    
    if not Path(test_image_path).exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    # Load image
    image = cv2.imread(test_image_path)
    print(f"Loaded image: {image.shape}")
    
    # Define objects to search for
    object_names = ["person", "car", "bicycle", "motorcycle", "bus", 
                   "truck", "bottle", "phone", "laptop", "bag"]
    
    print(f"Searching for: {', '.join(object_names)}")
    
    # Initialize pipeline with blob detection (faster for demo)
    pipeline = ObjectDetectionPipeline(
        proposal_method="blob_detection",
        confidence_threshold=0.05,
        use_soft_nms=False
    )
    
    # Run detection
    detections, debug_info = pipeline.detect_objects(image, object_names, debug=True)
    
    # Print results
    print(f"\nResults:")
    print(f"- Proposals generated: {debug_info['proposal_count']}")
    print(f"- Confident classifications: {debug_info['classification_count']}")
    print(f"- Final detections: {debug_info['final_count']}")
    print(f"- Total time: {debug_info['timings']['total']:.3f}s")
    
    print(f"\nDetected objects:")
    for i, (x, y, w, h, label, conf) in enumerate(detections):
        print(f"  {i+1}. {label}: {conf:.3f} at ({x},{y}) size {w}x{h}")
    
    return detections, debug_info

def demo_task2():
    """Demonstrate Task 2 - Barcode Analysis."""
    print("\\n=== TASK 2 DEMO: Barcode Analysis ===")
    
    # Test multiple images to find barcodes
    test_images = [
        "../Images/input_1.jpg",
        "../Images/input_2.jpg", 
        "../Images/input_3.jpg"
    ]
    
    # Initialize pipeline
    pipeline = BarcodeAnalysisPipeline(
        detection_params={
            "min_area": 500,
            "max_area": 50000,
            "aspect_ratio_range": (2.0, 12.0)
        },
        pose_params={
            "barcode_real_size": (40.0, 12.0)
        }
    )
    
    all_results = []
    
    for img_path in test_images:
        if not Path(img_path).exists():
            continue
            
        print(f"\\nTesting: {img_path}")
        image = cv2.imread(img_path)
        
        if image is None:
            continue
            
        # Run analysis
        results, debug_info = pipeline.analyze_barcode(image, debug=True)
        
        print(f"Results for {Path(img_path).name}:")
        print(f"- Candidate regions: {debug_info['detection_count']}")
        print(f"- Successful decodings: {debug_info['successful_decodings']}")
        print(f"- Successful pose estimations: {debug_info['successful_pose_estimations']}")
        
        for result in results:
            print(f"  Region {result['id']+1}:")
            print(f"    Decoded: {result['decoded_text'] or 'Failed'}")
            if result['normal_vector']:
                nv = result['normal_vector']
                print(f"    Normal: [{nv[0]:.3f}, {nv[1]:.3f}, {nv[2]:.3f}]")
            print(f"    Confidence: {result['confidence_scores'].get('overall', 0):.3f}")
        
        all_results.extend(results)
        
        # Only process first image with results for demo
        if results:
            break
    
    return all_results

def demo_integration():
    """Demonstrate integration of both tasks."""
    print("\\n=== INTEGRATION DEMO: Both Tasks ===")
    
    # Use main script functionality
    from main import process_single_image
    import argparse
    
    # Create mock arguments
    class Args:
        both = True
        image = "../Images/input_1.jpg" 
        objects = "person,car,phone,bottle"
        confidence = 0.1
        nms_threshold = 0.5
        barcode_size = "50,15"
        output = "../results"
        visualize = True
        export_json = True
        debug = False
        proposal_method = "blob_detection"
    
    args = Args()
    
    # Ensure output directory exists
    Path(args.output).mkdir(exist_ok=True)
    
    # Process image with both tasks
    if Path(args.image).exists():
        process_single_image(args.image, args)
    else:
        print(f"Demo image not found: {args.image}")

def run_performance_benchmark():
    """Run performance benchmarks on the pipelines."""
    print("\\n=== PERFORMANCE BENCHMARK ===")
    
    test_image_path = "../Images/input_1.jpg"
    if not Path(test_image_path).exists():
        print("Test image not found for benchmark")
        return
    
    image = cv2.imread(test_image_path)
    
    # Task 1 benchmark
    print("Task 1 Performance:")
    pipeline1 = ObjectDetectionPipeline(proposal_method="blob_detection")
    
    import time
    start_time = time.time()
    for _ in range(3):  # Run 3 times
        detections, _ = pipeline1.detect_objects(image, ["person", "car"], debug=False)
    avg_time = (time.time() - start_time) / 3
    print(f"  Average time per run: {avg_time:.3f}s")
    print(f"  Average detections: {len(detections)}")
    
    # Task 2 benchmark
    print("\\nTask 2 Performance:")
    pipeline2 = BarcodeAnalysisPipeline()
    
    start_time = time.time()
    for _ in range(3):  # Run 3 times  
        results, _ = pipeline2.analyze_barcode(image, debug=False)
    avg_time = (time.time() - start_time) / 3
    print(f"  Average time per run: {avg_time:.3f}s")
    print(f"  Average barcode regions: {len(results)}")

if __name__ == "__main__":
    """Run all demos."""
    print("🤖 Robotics Challenge - Computer Vision Demo")
    print("=" * 60)
    
    try:
        # Run demos
        demo_task1()
        demo_task2() 
        demo_integration()
        run_performance_benchmark()
        
        print("\\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()