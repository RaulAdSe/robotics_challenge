# Test Images

This directory contains test images for validation:

## Task 1 - Object Detection
- `test_objects_*.jpg` - Images with common objects (mugs, bottles, phones, etc.)
- `test_multi_*.jpg` - Images with multiple objects
- `test_edge_*.jpg` - Edge cases (small objects, partial occlusion)

## Task 2 - Barcode Detection  
- `test_barcode_*.jpg` - Images with Code128 barcodes
- `test_barcode_angled_*.jpg` - Barcodes at various angles
- `test_barcode_lighting_*.jpg` - Different lighting conditions

## Usage
Place your test images here following the naming convention above.
The test suite will automatically discover and use them for validation.