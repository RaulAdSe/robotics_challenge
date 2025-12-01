# Robotics Challenge - Computer Vision Tasks

## Overview
This project implements solutions for two computer vision tasks:
1. **Task 1**: Object detection with arbitrary text input (no pretrained detection models)
2. **Task 2**: Barcode detection, decoding, and 3D normal vector estimation (no ready-made libraries)

## Architecture
- **Task 1**: Selective Search + CLIP zero-shot classification
- **Task 2**: Gradient-based detection + manual Code128 decoding + PnP pose estimation

## Setup
```bash
pip install -r requirements.txt
```

## Project Structure
```
src/
├── task1_object_detection/    # Object detection pipeline
├── task2_barcode/            # Barcode detection and decoding
└── utils/                    # Shared utilities
tests/                        # Unit and integration tests
experiments/                  # Jupyter notebooks for testing
```

## Usage
```bash
python main.py --task1 --image path/to/image.jpg --objects "mug,bottle,phone"
python main.py --task2 --image path/to/barcode.jpg
```