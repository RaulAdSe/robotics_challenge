#!/usr/bin/env python3
"""
Main entry point for the Robotics Challenge Computer Vision Tasks.

This script provides a unified interface for both Task 1 (Object Detection) 
and Task 2 (Barcode Analysis) as specified in the challenge requirements.
"""

import argparse
import cv2
import numpy as np
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from task1_object_detection.pipeline import ObjectDetectionPipeline
from task2_barcode.pipeline import BarcodeAnalysisPipeline
from utils.relationship_matcher import match_objects_to_barcodes, query_relationship, get_relationship_summary

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robotics Challenge - Computer Vision Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Task 1: Object detection with text input
  python main.py --task1 --image Images/input_1.jpg --objects "person,car,bottle"
  
  # Task 2: Barcode detection and analysis  
  python main.py --task2 --image Images/input_2.jpg
  
  # Both tasks on the same image
  python main.py --both --image Images/input_3.jpg --objects "phone,laptop"
  
  # Batch processing
  python main.py --task1 --batch Images/ --objects "person,car,phone" --output results/
        """
    )
    
    # Task selection (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task1", action="store_true", 
                           help="Run Task 1: Object detection with text input")
    task_group.add_argument("--task2", action="store_true",
                           help="Run Task 2: Barcode detection and analysis")
    task_group.add_argument("--both", action="store_true",
                           help="Run both tasks on the same image")
    
    # Input specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str,
                            help="Path to input image file")
    input_group.add_argument("--batch", type=str,
                            help="Path to directory for batch processing")
    
    # Task 1 specific arguments
    parser.add_argument("--objects", type=str,
                       help="Comma-separated list of objects to detect (required for Task 1)")
    parser.add_argument("--confidence", type=float, default=0.1,
                       help="Confidence threshold for detections (default: 0.1)")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                       help="NMS IoU threshold (default: 0.5)")
    
    # Task 2 specific arguments  
    parser.add_argument("--barcode-size", type=str, default="50,15",
                       help="Real-world barcode size in mm as 'width,height' (default: 50,15)")
    
    # Output options
    parser.add_argument("--output", type=str, default=".",
                       help="Output directory for results (default: current directory)")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization images")
    parser.add_argument("--export-json", action="store_true", 
                       help="Export results to JSON")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed output")
    
    # Task 3: Relationship queries
    parser.add_argument("--query", type=str,
                       help="Query object-barcode relationship (e.g. 'bottle' or barcode text)")
    parser.add_argument("--show-relationships", action="store_true",
                       help="Display all object-barcode relationships")
    
    # Performance options
    parser.add_argument("--proposal-method", type=str, default="blob_detection",
                       choices=["selective_search", "blob_detection", "both"],
                       help="Region proposal method (default: blob_detection)")
    
    return parser.parse_args()

def load_image(image_path: str) -> np.ndarray:
    """Load and validate image file."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image

def run_task1(image: np.ndarray, 
              object_names: List[str],
              args) -> tuple[List, Dict]:
    """Run Task 1: Object detection with text input."""
    print("🔍 Running Task 1: Object Detection with Text Input")
    print(f"Looking for objects: {', '.join(object_names)}")
    
    # Parse barcode size
    barcode_size = [float(x.strip()) for x in args.barcode_size.split(',')]
    
    # Initialize pipeline
    pipeline = ObjectDetectionPipeline(
        proposal_method=args.proposal_method,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold,
        use_soft_nms=False
    )
    
    # Run detection
    detections, debug_info = pipeline.detect_objects(image, object_names, debug=args.debug)
    
    # Print results
    print(f"✅ Found {len(detections)} objects:")
    for i, (x, y, w, h, label, conf) in enumerate(detections):
        print(f"  {i+1}. {label}: {conf:.3f} confidence at ({x},{y}) size {w}x{h}")
    
    return detections, debug_info

def run_task2(image: np.ndarray, args) -> tuple[List[Dict], Dict]:
    """Run Task 2: Barcode detection and analysis."""
    print("📱 Running Task 2: Barcode Detection and Analysis")
    
    # Parse barcode size
    barcode_size = [float(x.strip()) for x in args.barcode_size.split(',')]
    
    # Initialize pipeline
    pipeline = BarcodeAnalysisPipeline(
        detection_params={
            "min_area": 500,
            "max_area": 100000,
            "aspect_ratio_range": (2.0, 15.0)
        },
        pose_params={
            "barcode_real_size": tuple(barcode_size)
        }
    )
    
    # Run analysis
    results, debug_info = pipeline.analyze_barcode(image, debug=args.debug)
    
    # Print results
    print(f"✅ Analyzed {len(results)} barcode regions:")
    for result in results:
        decoded_text = result.get('decoded_text')
        normal_vector = result.get('normal_vector')
        confidence = result.get('confidence_scores', {}).get('overall', 0.0)
        
        print(f"  Region {result['id']+1}:")
        print(f"    Text: {decoded_text if decoded_text else 'Failed to decode'}")
        if normal_vector:
            print(f"    Normal: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")
        print(f"    Confidence: {confidence:.3f}")
    
    return results, debug_info

def save_results(image_path: str,
                task1_results=None, 
                task2_results=None,
                task1_debug=None,
                task2_debug=None,
                args=None,
                associations=None):
    """Save results and visualizations."""
    base_name = Path(image_path).stem
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load original image for visualization
    image = load_image(image_path)
    
    # Save Task 1 results
    if task1_results is not None:
        if args.visualize:
            # Import visualization utilities
            from src.utils.visualization import draw_bounding_boxes
            
            # Draw detections
            boxes = [(x, y, w, h) for x, y, w, h, _, _ in task1_results]
            labels = [f"{label} ({conf:.2f})" for _, _, _, _, label, conf in task1_results]
            
            result_img = draw_bounding_boxes(image, boxes, labels)
            
            viz_path = output_dir / f"{base_name}_task1_result.jpg"
            cv2.imwrite(str(viz_path), result_img)
            print(f"Task 1 visualization saved: {viz_path}")
        
        if args.export_json:
            # Convert detections to JSON-serializable format
            detections_json = []
            for x, y, w, h, label, conf in task1_results:
                detections_json.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "label": label,
                    "confidence": float(conf)
                })
            
            json_path = output_dir / f"{base_name}_task1_results.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "task": "object_detection",
                    "image_path": image_path,
                    "detections": detections_json,
                    "total_detections": len(task1_results)
                }, f, indent=2)
            print(f"Task 1 results exported: {json_path}")
    
    # Save Task 2 results
    if task2_results is not None:
        if args.visualize:
            from src.task2_barcode.pipeline import BarcodeAnalysisPipeline
            
            # Create pipeline for visualization
            pipeline = BarcodeAnalysisPipeline()
            result_img = pipeline.visualize_results(image, task2_results)
            
            viz_path = output_dir / f"{base_name}_task2_result.jpg"
            cv2.imwrite(str(viz_path), result_img)
            print(f"Task 2 visualization saved: {viz_path}")
        
        if args.export_json:
            json_path = output_dir / f"{base_name}_task2_results.json"
            with open(json_path, 'w') as f:
                # Clean results for JSON export
                clean_results = []
                for result in task2_results:
                    clean_result = {
                        "id": result["id"],
                        "corners": result["corners"],
                        "decoded_text": result["decoded_text"],
                        "normal_vector": result["normal_vector"],
                        "confidence_scores": result["confidence_scores"]
                    }
                    clean_results.append(clean_result)
                
                json.dump({
                    "task": "barcode_analysis", 
                    "image_path": image_path,
                    "barcode_results": clean_results,
                    "total_barcodes": len(task2_results)
                }, f, indent=2)
            print(f"Task 2 results exported: {json_path}")
    
    # Save Task 3 results (relationships)
    if associations is not None and args.export_json:
        json_path = output_dir / f"{base_name}_task3_relationships.json"
        with open(json_path, 'w') as f:
            json.dump({
                "task": "object_barcode_relationships",
                "image_path": image_path,
                "relationships": associations,
                "total_matches": len(associations["matches"]),
                "summary": get_relationship_summary(associations)
            }, f, indent=2)
        print(f"Task 3 relationships exported: {json_path}")

def process_single_image(image_path: str, args):
    """Process a single image with specified tasks."""
    print(f"\n📸 Processing: {image_path}")
    print(f"Image size: {cv2.imread(image_path).shape}")
    
    try:
        # Load image
        image = load_image(image_path)
        
        task1_results = None
        task2_results = None
        task1_debug = None
        task2_debug = None
        
        # Run Task 1 if requested
        if args.task1 or args.both:
            if not args.objects:
                print("❌ Error: --objects required for Task 1")
                return
            
            object_names = [obj.strip() for obj in args.objects.split(',')]
            task1_results, task1_debug = run_task1(image, object_names, args)
        
        # Run Task 2 if requested  
        if args.task2 or args.both:
            task2_results, task2_debug = run_task2(image, args)
        
        # Run Task 3: Object-Barcode relationship matching
        associations = None
        if task1_results and task2_results:
            print("\n🔗 Running Task 3: Object-Barcode Relationship Analysis")
            associations = match_objects_to_barcodes(task1_results, task2_results)
            
            if args.show_relationships or args.debug:
                relationship_summary = get_relationship_summary(associations)
                print(relationship_summary)
            
            # Handle query if provided
            if args.query:
                result = query_relationship(associations, args.query)
                if result:
                    print(f"\n🔍 Query Result: '{args.query}' → '{result}'")
                else:
                    print(f"\n❌ No relationship found for: '{args.query}'")
        
        # Save results
        save_results(image_path, task1_results, task2_results, 
                    task1_debug, task2_debug, args, associations)
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def process_batch(batch_dir: str, args):
    """Process all images in a directory."""
    batch_path = Path(batch_dir)
    if not batch_path.is_dir():
        raise NotADirectoryError(f"Batch directory not found: {batch_dir}")
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(batch_path.glob(f'*{ext}'))
        image_files.extend(batch_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No image files found in {batch_dir}")
        return
    
    print(f"🔄 Processing {len(image_files)} images from {batch_dir}")
    
    # Process each image
    for i, image_file in enumerate(sorted(image_files)):
        print(f"\n[{i+1}/{len(image_files)}]", end=" ")
        process_single_image(str(image_file), args)

def main():
    """Main entry point."""
    args = parse_arguments()
    
    print("🤖 Robotics Challenge - Computer Vision Tasks")
    print("=" * 50)
    
    # Validate arguments
    if (args.task1 or args.both) and not args.objects:
        print("❌ Error: --objects is required when using --task1 or --both")
        sys.exit(1)
    
    try:
        # Process images
        if args.image:
            process_single_image(args.image, args)
        elif args.batch:
            process_batch(args.batch, args)
        
        print("\n✅ Processing completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()