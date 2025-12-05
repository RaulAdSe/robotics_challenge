#!/usr/bin/env python3
"""
Barcode Detection Parameter Tuning Script

Evaluates different parameter configurations against ground truth.
Ground truth: images 1-4 have 5 barcodes, images 5-6 have 3, images 7-8 have 5.

Usage:
    python scripts/tune_barcode_detection.py
    python scripts/tune_barcode_detection.py --quick   # Fast mode with fewer configs
    python scripts/tune_barcode_detection.py --verbose # Show per-image details
"""

import cv2
import numpy as np
import sys
import os
import argparse
import itertools
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'task2_barcode'))

from detection_bw import BlackWhiteBarcodeDetector
from decoder import Code128Decoder

# Ground truth
GROUND_TRUTH = {
    1: 5, 2: 5, 3: 5, 4: 5,  # images 1-4: 5 barcodes each
    5: 3, 6: 3,               # images 5-6: 3 barcodes each
    7: 5, 8: 5                # images 7-8: 5 barcodes each
}
TOTAL_EXPECTED = sum(GROUND_TRUTH.values())  # 36 total


def evaluate_config(params, images, decoder, verbose=False):
    """Evaluate a parameter configuration."""
    detector = BlackWhiteBarcodeDetector(**params)

    total_detected = 0
    total_decoded = 0
    total_true_positives = 0  # min(detected, expected)
    total_false_positives = 0  # max(0, detected - expected)
    total_false_negatives = 0  # max(0, expected - detected)

    per_image = []

    for img_num, image in images.items():
        expected = GROUND_TRUTH[img_num]

        detections, _ = detector.detect_barcodes(image)
        detected = len(detections)

        # Count decoded
        decoded = 0
        for det in detections:
            if det['orientation'] == 'vertical':
                rectified = detector.extract_barcode_region(image, det, output_size=(100, 300))
                rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)
            else:
                rectified = detector.extract_barcode_region(image, det, output_size=(300, 100))

            text, _ = decoder.decode_barcode(rectified)
            if text:
                decoded += 1

        total_detected += detected
        total_decoded += decoded

        # Calculate metrics
        tp = min(detected, expected)
        fp = max(0, detected - expected)
        fn = max(0, expected - detected)

        total_true_positives += tp
        total_false_positives += fp
        total_false_negatives += fn

        per_image.append({
            'img': img_num,
            'expected': expected,
            'detected': detected,
            'decoded': decoded,
            'tp': tp, 'fp': fp, 'fn': fn
        })

    # Calculate overall metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    decode_rate = total_decoded / total_detected if total_detected > 0 else 0

    return {
        'detected': total_detected,
        'decoded': total_decoded,
        'expected': TOTAL_EXPECTED,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'decode_rate': decode_rate,
        'per_image': per_image
    }


def generate_param_configs(quick=False):
    """Generate parameter configurations to test."""
    if quick:
        # Quick mode: fewer combinations
        configs = [
            # Baseline
            {'min_aspect': 1.3, 'min_area_pct': 0.0003, 'min_solidity': 0.5,
             'min_transitions': 10, 'min_contrast': 40, 'min_score': 0.5},
            # More permissive
            {'min_aspect': 1.2, 'min_area_pct': 0.0002, 'min_solidity': 0.4,
             'min_transitions': 8, 'min_contrast': 30, 'min_score': 0.4},
            # More strict
            {'min_aspect': 1.5, 'min_area_pct': 0.0005, 'min_solidity': 0.55,
             'min_transitions': 12, 'min_contrast': 50, 'min_score': 0.6},
            # Balanced
            {'min_aspect': 1.4, 'min_area_pct': 0.00025, 'min_solidity': 0.45,
             'min_transitions': 10, 'min_contrast': 35, 'min_score': 0.5},
        ]
    else:
        # Full grid search
        param_grid = {
            'min_aspect': [1.2, 1.3, 1.4, 1.5],
            'min_area_pct': [0.0002, 0.0003, 0.0004],
            'min_solidity': [0.4, 0.5, 0.55],
            'min_transitions': [8, 10, 12],
            'min_contrast': [30, 40, 50],
            'min_score': [0.4, 0.5, 0.6],
        }

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    return configs


def main():
    parser = argparse.ArgumentParser(description='Tune barcode detection parameters')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer configs')
    parser.add_argument('--verbose', action='store_true', help='Show per-image details')
    parser.add_argument('--top', type=int, default=10, help='Show top N results')
    args = parser.parse_args()

    print("=" * 80)
    print("BARCODE DETECTION PARAMETER TUNING")
    print("=" * 80)
    print(f"\nGround Truth: {TOTAL_EXPECTED} total barcodes")
    print(f"  Images 1-4: 5 each, Images 5-6: 3 each, Images 7-8: 5 each\n")

    # Load images
    print("Loading images...")
    images = {}
    for img_num in range(1, 9):
        img_path = f'Images/input_{img_num}.jpg'
        image = cv2.imread(img_path)
        if image is not None:
            images[img_num] = image
        else:
            print(f"  WARNING: Could not load {img_path}")

    # Initialize decoder
    decoder = Code128Decoder(num_scanlines=9, tolerance=0.4)

    # Generate configs
    configs = generate_param_configs(quick=args.quick)
    print(f"\nTesting {len(configs)} parameter configurations...")
    print("-" * 80)

    results = []

    for i, params in enumerate(configs):
        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(configs)}", end='\r')

        metrics = evaluate_config(params, images, decoder, verbose=args.verbose)
        results.append({
            'params': params,
            'metrics': metrics
        })

    print(f"  Progress: {len(configs)}/{len(configs)} - Done!")

    # Sort by F1 score (best balance of precision and recall)
    results.sort(key=lambda x: x['metrics']['f1'], reverse=True)

    # Display top results
    print("\n" + "=" * 80)
    print(f"TOP {args.top} CONFIGURATIONS (sorted by F1 score)")
    print("=" * 80)

    for rank, result in enumerate(results[:args.top], 1):
        params = result['params']
        m = result['metrics']

        print(f"\n{'─' * 80}")
        print(f"RANK #{rank}")
        print(f"{'─' * 80}")
        print(f"  Detected: {m['detected']}/{m['expected']} | Decoded: {m['decoded']} ({m['decode_rate']*100:.0f}%)")
        print(f"  Precision: {m['precision']*100:.1f}% | Recall: {m['recall']*100:.1f}% | F1: {m['f1']*100:.1f}%")
        print(f"  Parameters:")
        print(f"    min_aspect={params['min_aspect']}, min_area_pct={params['min_area_pct']}")
        print(f"    min_solidity={params['min_solidity']}, min_transitions={params['min_transitions']}")
        print(f"    min_contrast={params['min_contrast']}, min_score={params['min_score']}")

        if args.verbose:
            print(f"  Per-image breakdown:")
            for pi in m['per_image']:
                status = "✓" if pi['detected'] == pi['expected'] else ("↑" if pi['detected'] > pi['expected'] else "↓")
                print(f"    input_{pi['img']}: {pi['detected']}/{pi['expected']} detected, {pi['decoded']} decoded {status}")

    # Best configuration summary
    best = results[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"""
To use this configuration, update detection_bw.py or pass these parameters:

detector = BlackWhiteBarcodeDetector(
    min_aspect={best['params']['min_aspect']},
    min_area_pct={best['params']['min_area_pct']},
    min_solidity={best['params']['min_solidity']},
    min_transitions={best['params']['min_transitions']},
    min_contrast={best['params']['min_contrast']},
    min_score={best['params']['min_score']}
)

Expected results:
  - Detection: {best['metrics']['detected']}/{best['metrics']['expected']} ({best['metrics']['recall']*100:.0f}% recall)
  - Decoding: {best['metrics']['decoded']} ({best['metrics']['decode_rate']*100:.0f}% decode rate)
  - F1 Score: {best['metrics']['f1']*100:.1f}%
""")

    # Save results to file
    output_file = f"Out/task2_barcode_v2/tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("BARCODE DETECTION PARAMETER TUNING RESULTS\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Configurations tested: {len(configs)}\n\n")

        for rank, result in enumerate(results[:20], 1):
            params = result['params']
            m = result['metrics']
            f.write(f"Rank {rank}: F1={m['f1']*100:.1f}%, P={m['precision']*100:.1f}%, R={m['recall']*100:.1f}%\n")
            f.write(f"  Detected: {m['detected']}/{m['expected']}, Decoded: {m['decoded']}\n")
            f.write(f"  {params}\n\n")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
