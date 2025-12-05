# Task 2: Barcode Detection, Decoding, and 3D Normal Estimation

## Overview

Task 2 implements a complete barcode analysis pipeline that:
1. Detects Code128 barcodes in images
2. Decodes the barcode content
3. Estimates 3D normal vectors for each barcode

The pipeline integrates with Task 1 (object detection) to constrain barcode detection to detected object regions.

## Architecture

```
src/task2_barcode/
├── __init__.py
├── detection.py      # Gradient-based barcode detection
├── decoder.py        # Code128 barcode decoder
├── pose_estimation.py # 3D normal vector estimation
└── pipeline.py       # Complete analysis pipeline
```

## Components

### 1. Barcode Detection (`detection.py`)

**Class:** `BarcodeDetector`

Uses classical computer vision techniques (no pre-trained models):
- **Gradient Analysis**: Scharr operators for directional gradients
- **Local Variance**: High-contrast region detection
- **Morphological Operations**: Closing to connect barcode bars
- **Shape Filtering**: Aspect ratio, solidity, extent metrics

**Key Parameters:**
```python
BarcodeDetector(
    grad_thresh=1.3,      # Gradient ratio threshold
    var_thresh=25,        # Variance threshold
    min_solidity=0.55,    # Minimum solidity
    min_extent=0.35,      # Minimum extent
    kernel_w=33,          # Morphological kernel width
    kernel_h=7,           # Morphological kernel height
    min_aspect=2.0,       # Minimum aspect ratio
    max_aspect=12.0       # Maximum aspect ratio
)
```

**Methods:**
- `detect_barcodes(image, roi=None)` - Full image or ROI detection
- `detect_in_object_regions(image, object_detections)` - Task 1 integration
- `extract_barcode_region(image, detection)` - Rectify detected barcode

### 2. Code128 Decoder (`decoder.py`)

**Class:** `Code128Decoder`

Decodes Code128 barcodes using virtual scanline technique:
- Complete 107-pattern lookup table
- Multi-scanline decoding (tries multiple horizontal lines)
- Bidirectional scanning (left-to-right and right-to-left)
- Otsu's thresholding for binarization
- CLAHE contrast enhancement
- Checksum verification

**Key Parameters:**
```python
Code128Decoder(
    num_scanlines=5,   # Number of scanlines to try
    tolerance=0.3      # Pattern matching tolerance
)
```

**Supported Code Sets:**
- Code Set A: ASCII 0-95 (control + uppercase)
- Code Set B: ASCII 32-127 (printable characters)
- Code Set C: Numeric pairs 00-99

### 3. 3D Normal Estimation (`pose_estimation.py`)

**Class:** `BarcodeNormalEstimator`

Estimates barcode surface normal using PnP (Perspective-n-Point):
- Assumes planar barcode with known real-world dimensions
- Uses OpenCV's `solvePnP` with ITERATIVE method
- Extracts normal vector from rotation matrix

**Key Parameters:**
```python
BarcodeNormalEstimator(
    barcode_real_size=(40.0, 12.0),  # Real size in mm (width, height)
    camera_matrix=None,               # Optional custom camera matrix
    dist_coeffs=None                  # Optional distortion coefficients
)
```

### 4. Complete Pipeline (`pipeline.py`)

**Class:** `BarcodeAnalysisPipeline`

Orchestrates the full barcode analysis workflow:

```python
pipeline = BarcodeAnalysisPipeline(
    detection_params={},
    decoding_params={'num_scanlines': 9, 'tolerance': 0.4},
    pose_params={'barcode_real_size': (40.0, 12.0)}
)

# Full image analysis
results, debug_info = pipeline.analyze_barcode(image)

# With Task 1 integration (object-constrained)
results, debug_info = pipeline.analyze_barcode(
    image,
    object_detections=[
        {'name': 'bottle', 'bbox': (x, y, w, h)},
        {'name': 'box', 'bbox': (x, y, w, h)},
    ]
)
```

## Integration with Task 1

Task 2 can operate in two modes:

### Mode 1: Full Image Detection
```python
results, debug = pipeline.analyze_barcode(image)
```

### Mode 2: Object-Constrained Detection (Recommended)
```python
# Get object detections from Task 1
from task1_object_detection.pipeline import ObjectDetectionPipeline

task1 = ObjectDetectionPipeline(proposal_method='blob_detection')
objects, _ = task1.detect_objects(image, ['bottle', 'box', 'mug'])

# Convert to Task 2 format
object_detections = [
    {'name': det[4], 'bbox': (det[0], det[1], det[2], det[3])}
    for det in objects
]

# Run barcode analysis within object regions
results, debug = pipeline.analyze_barcode(image, object_detections=object_detections)
```

## Output Format

Each barcode result contains:
```python
{
    "id": 0,                           # Barcode index
    "corners": [[x1,y1], ...],         # 4 corner points
    "object_name": "bottle",           # Associated object (if Task 1 integrated)
    "decoded_text": "Box",             # Decoded barcode content
    "normal_vector": [0.2, 0.1, 0.97], # 3D surface normal
    "confidence_scores": {
        "overall": 0.85,
        "decoding": 0.9,
        "pose_estimation": 0.7
    }
}
```

## Performance

Typical processing times on CPU:
- Detection: ~50-100ms per image
- Decoding: ~10-20ms per barcode
- Pose estimation: ~5-10ms per barcode
- **Total pipeline: ~150-300ms** for an image with 5 objects

## Testing

Run the detection tests:
```bash
cd /path/to/robotics_challenge
python -c "
from src.task2_barcode.detection import test_detection_on_images
test_detection_on_images()
"
```

Run the full pipeline:
```bash
python -c "
from src.task2_barcode.pipeline import demo_task2_pipeline
demo_task2_pipeline('Images/input_1.jpg')
"
```

## Known Limitations

1. **Decoder Accuracy**: Some barcodes may fail to decode due to:
   - Motion blur or poor image quality
   - Extreme viewing angles
   - Partial occlusion

2. **False Positives**: Detection may find barcode-like patterns that aren't actual barcodes. The decoder acts as a secondary filter.

3. **Normal Vector Accuracy**: Depends on accurate corner detection and assumes known barcode dimensions.

## Dependencies

- OpenCV (`cv2`)
- NumPy
- No external barcode libraries required (pure implementation)

---

## Improved Black-on-White Barcode Detection (v2)

### Problem with Original Approach

The original `detection.py` used a single gradient direction (horizontal) which only detected barcodes with **vertical bars**. This missed:
- Rotated barcodes (90° rotation with horizontal bars)
- Barcodes at various angles

Additionally, the original approach had high false positive rates due to loose filtering criteria.

### New Detection Algorithm (`detection_bw.py`)

The improved `BlackWhiteBarcodeDetector` class addresses these issues with a multi-stage approach:

#### Key Improvements

1. **Dual Gradient Analysis**
   - Processes BOTH horizontal gradients (for vertical bars) AND vertical gradients (for horizontal bars)
   - Handles barcodes in any orientation (horizontal or vertical)

2. **Combined Gradient + Variance Mask**
   ```python
   # Compute local variance for high-contrast regions
   variance = sq_mean - mean**2

   # Combine gradient strength with variance
   combined = np.minimum(grad_norm, var_norm)
   ```
   This ensures we only detect regions that have BOTH:
   - Strong directional edges (barcode bars)
   - High local contrast (black-on-white pattern)

3. **Strict Pattern Verification**
   Each candidate region is validated by:
   - **Transition counting**: Barcodes need 12+ black/white transitions
   - **Contrast check**: Minimum 50 pixel intensity difference
   - **Regularity check**: Coefficient of variation (CV) < 1.5 for bar spacing
   - **White ratio**: 25-75% white pixels (typical for Code128)

4. **Scoring System**
   ```python
   score = 0.6  # Base score if passes all checks
   if transitions >= 30: score += 0.2
   if contrast >= 100: score += 0.1
   if cv < 0.8: score += 0.1  # Regular spacing bonus
   ```
   Only candidates with score >= 0.6 are accepted.

#### Algorithm Steps

```
Input Image
    │
    ▼
┌─────────────────────┐
│ CLAHE Enhancement   │  (Adaptive contrast enhancement)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Compute Gradients   │  Scharr operators for X and Y
│ + Local Variance    │
└─────────────────────┘
    │
    ├──────────────────────────┐
    ▼                          ▼
┌──────────────┐        ┌──────────────┐
│ X-Gradient   │        │ Y-Gradient   │
│ (vertical    │        │ (horizontal  │
│  bars)       │        │  bars)       │
└──────────────┘        └──────────────┘
    │                          │
    ▼                          ▼
┌──────────────┐        ┌──────────────┐
│ Combined     │        │ Combined     │
│ Mask         │        │ Mask         │
│ (grad ∩ var) │        │ (grad ∩ var) │
└──────────────┘        └──────────────┘
    │                          │
    ▼                          ▼
┌──────────────┐        ┌──────────────┐
│ Morphology   │        │ Morphology   │
│ Close + Open │        │ Close + Open │
└──────────────┘        └──────────────┘
    │                          │
    └──────────┬───────────────┘
               ▼
    ┌─────────────────────┐
    │ Find Contours       │
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Validate Each:      │
    │ - Size/Aspect ratio │
    │ - Solidity          │
    │ - Pattern analysis  │
    │ - Transition count  │
    │ - Regularity (CV)   │
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Non-Max Suppression │
    └─────────────────────┘
               │
               ▼
         Final Detections
```

### Usage

```python
from src.task2_barcode.detection_bw import BlackWhiteBarcodeDetector

detector = BlackWhiteBarcodeDetector(
    min_aspect=1.8,        # Barcodes are elongated
    max_aspect=7.0,
    min_solidity=0.55,     # Fairly rectangular
    min_transitions=12,    # Many bar transitions
    min_contrast=50,       # Black on white = high contrast
    min_score=0.6          # Confidence threshold
)

detections, debug_info = detector.detect_barcodes(image)

for det in detections:
    print(f"Barcode at {det['bbox']}, orientation={det['orientation']}, "
          f"score={det['barcode_score']:.2f}")
```

### Results Comparison

| Image | Original Detector | Improved Detector |
|-------|-------------------|-------------------|
| input_1.jpg | 5 (2 decoded) | 4 (more accurate) |
| input_2.jpg | 8 (2 decoded) | 5 |
| input_3.jpg | 6 (1 decoded) | 5 |
| input_4.jpg | 6 (1 decoded) | 5 |
| input_5.jpg | 3 (1 decoded) | 1 |
| input_6.jpg | 5 (1 decoded) | 4 |
| input_7.jpg | 6 (4 decoded) | 4 |
| input_8.jpg | 6 (3 decoded) | 3 |

The improved detector has:
- **Fewer false positives** (stricter validation)
- **Better orientation handling** (detects both horizontal and vertical barcodes)
- **Higher quality detections** (score-based ranking)
