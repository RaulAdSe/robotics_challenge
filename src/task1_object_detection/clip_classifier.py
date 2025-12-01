"""CLIP-based zero-shot classification for region proposals."""

import torch
import clip
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import time

class CLIPClassifier:
    """Zero-shot object classifier using CLIP."""
    
    def __init__(self, model_name: str = "ViT-B/32", confidence_threshold: float = 0.1):
        """
        Initialize CLIP classifier.
        
        Args:
            model_name: CLIP model variant to use
            confidence_threshold: Minimum confidence for positive classification
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print(f"CLIP model loaded successfully!")
    
    def classify_regions(self, 
                        image: np.ndarray,
                        regions: List[Tuple[int, int, int, int]],
                        text_prompts: List[str]) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Classify image regions against text prompts.
        
        Args:
            image: Input image (BGR format)
            regions: List of bounding boxes as (x, y, w, h)
            text_prompts: List of object names to classify against
            
        Returns:
            List of (x, y, w, h, label, confidence) for regions above threshold
        """
        if not regions or not text_prompts:
            return []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare text inputs
        text_inputs = self._prepare_text_prompts(text_prompts)
        
        results = []
        
        # Process regions (can be batched for efficiency)
        for region in regions:
            x, y, w, h = region
            
            # Extract and validate region
            roi = self._extract_region(rgb_image, x, y, w, h)
            if roi is None:
                continue
            
            # Classify region
            label, confidence = self._classify_single_region(roi, text_inputs, text_prompts)
            
            if confidence >= self.confidence_threshold:
                results.append((x, y, w, h, label, confidence))
        
        return results
    
    def classify_regions_batch(self,
                             image: np.ndarray,
                             regions: List[Tuple[int, int, int, int]], 
                             text_prompts: List[str],
                             batch_size: int = 32) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Classify regions in batches for improved efficiency.
        
        Args:
            image: Input image (BGR format)
            regions: List of bounding boxes as (x, y, w, h)
            text_prompts: List of object names to classify against
            batch_size: Number of regions to process in parallel
            
        Returns:
            List of (x, y, w, h, label, confidence) for regions above threshold
        """
        if not regions or not text_prompts:
            return []
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare text inputs once
        text_inputs = self._prepare_text_prompts(text_prompts)
        
        results = []
        
        # Process in batches
        for i in range(0, len(regions), batch_size):
            batch_regions = regions[i:i + batch_size]
            batch_results = self._process_region_batch(rgb_image, batch_regions, text_inputs, text_prompts)
            results.extend(batch_results)
        
        return results
    
    def _prepare_text_prompts(self, text_prompts: List[str]) -> torch.Tensor:
        """Prepare text prompts for CLIP processing."""
        # Add "a photo of" prefix to improve performance
        formatted_prompts = [f"a photo of a {prompt}" for prompt in text_prompts]
        text_inputs = clip.tokenize(formatted_prompts).to(self.device)
        return text_inputs
    
    def _extract_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        """Extract and validate region from image."""
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        
        # Validate region size
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return None
        
        return roi
    
    def _classify_single_region(self, 
                               roi: np.ndarray,
                               text_inputs: torch.Tensor,
                               text_prompts: List[str]) -> Tuple[str, float]:
        """Classify a single region against text prompts."""
        try:
            # Convert to PIL Image and preprocess
            pil_image = Image.fromarray(roi)
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Compute features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarities[0].topk(1)
                
                confidence = values.item()
                best_match_idx = indices.item()
                label = text_prompts[best_match_idx]
                
                return label, confidence
                
        except Exception as e:
            print(f"Error classifying region: {e}")
            return "unknown", 0.0
    
    def _process_region_batch(self,
                            image: np.ndarray,
                            regions: List[Tuple[int, int, int, int]],
                            text_inputs: torch.Tensor,
                            text_prompts: List[str]) -> List[Tuple[int, int, int, int, str, float]]:
        """Process a batch of regions efficiently."""
        batch_results = []
        valid_regions = []
        valid_rois = []
        
        # Extract all valid ROIs
        for region in regions:
            x, y, w, h = region
            roi = self._extract_region(image, x, y, w, h)
            if roi is not None:
                valid_regions.append(region)
                valid_rois.append(roi)
        
        if not valid_rois:
            return []
        
        try:
            # Preprocess all images in batch
            image_inputs = []
            for roi in valid_rois:
                pil_image = Image.fromarray(roi)
                preprocessed = self.preprocess(pil_image)
                image_inputs.append(preprocessed)
            
            # Stack into batch
            image_batch = torch.stack(image_inputs).to(self.device)
            
            # Compute features for entire batch
            with torch.no_grad():
                image_features = self.model.encode_image(image_batch)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities for all images
                similarities = (image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarities.topk(1, dim=-1)
                
                # Extract results
                for i, (region, confidence_tensor, label_idx_tensor) in enumerate(zip(
                    valid_regions, values, indices)):
                    
                    confidence = confidence_tensor.item()
                    label_idx = label_idx_tensor.item()
                    label = text_prompts[label_idx]
                    
                    if confidence >= self.confidence_threshold:
                        x, y, w, h = region
                        batch_results.append((x, y, w, h, label, confidence))
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Fallback to individual processing
            for region in valid_regions:
                x, y, w, h = region
                roi = self._extract_region(image, x, y, w, h)
                if roi is not None:
                    label, confidence = self._classify_single_region(roi, text_inputs, text_prompts)
                    if confidence >= self.confidence_threshold:
                        batch_results.append((x, y, w, h, label, confidence))
        
        return batch_results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold
        }

def benchmark_clip_performance(image: np.ndarray, 
                             regions: List[Tuple[int, int, int, int]],
                             text_prompts: List[str]) -> Dict[str, float]:
    """Benchmark CLIP classification performance."""
    classifier = CLIPClassifier()
    
    # Test individual processing
    start_time = time.time()
    results_individual = classifier.classify_regions(image, regions[:10], text_prompts)
    individual_time = time.time() - start_time
    
    # Test batch processing  
    start_time = time.time()
    results_batch = classifier.classify_regions_batch(image, regions[:10], text_prompts, batch_size=5)
    batch_time = time.time() - start_time
    
    return {
        "individual_time": individual_time,
        "batch_time": batch_time,
        "individual_results": len(results_individual),
        "batch_results": len(results_batch),
        "speedup": individual_time / batch_time if batch_time > 0 else 0
    }

if __name__ == "__main__":
    # Test CLIP classifier
    import sys
    from pathlib import Path
    
    # Load test image
    image = cv2.imread("Images/input_1.jpg")
    if image is not None:
        print(f"Loaded test image: {image.shape}")
        
        # Create some test regions
        regions = [(100, 100, 200, 200), (300, 300, 150, 150)]
        text_prompts = ["person", "car", "dog", "bottle", "chair"]
        
        # Test classifier
        classifier = CLIPClassifier(confidence_threshold=0.05)
        results = classifier.classify_regions(image, regions, text_prompts)
        
        print(f"Classification results:")
        for x, y, w, h, label, confidence in results:
            print(f"  Region ({x},{y},{w},{h}): {label} ({confidence:.3f})")
    else:
        print("Could not load test image")