"""Tests for CLIP classifier (mock tests for structure validation)."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestCLIPClassifier:
    """Test cases for CLIP classifier."""
    
    @patch('task1_object_detection.clip_classifier.clip')
    @patch('task1_object_detection.clip_classifier.torch')
    def test_init(self, mock_torch, mock_clip):
        """Test CLIP classifier initialization."""
        # Mock the CLIP model loading
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)
        mock_torch.cuda.is_available.return_value = False
        
        from task1_object_detection.clip_classifier import CLIPClassifier
        
        classifier = CLIPClassifier()
        
        assert classifier.model_name == "ViT-B/32"
        assert classifier.confidence_threshold == 0.1
        assert classifier.device == "cpu"
        mock_clip.load.assert_called_once_with("ViT-B/32", device="cpu")
    
    @patch('task1_object_detection.clip_classifier.clip')
    @patch('task1_object_detection.clip_classifier.torch')
    def test_custom_params(self, mock_torch, mock_clip):
        """Test initialization with custom parameters."""
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)
        mock_torch.cuda.is_available.return_value = True
        
        from task1_object_detection.clip_classifier import CLIPClassifier
        
        classifier = CLIPClassifier(
            model_name="ViT-L/14", 
            confidence_threshold=0.25
        )
        
        assert classifier.model_name == "ViT-L/14"
        assert classifier.confidence_threshold == 0.25
        assert classifier.device == "cuda"
    
    def test_extract_region(self):
        """Test region extraction validation."""
        # We can test this method without mocking since it's pure OpenCV/NumPy
        from task1_object_detection.clip_classifier import CLIPClassifier
        
        # Create mock classifier (we'll test the method directly)
        classifier = Mock(spec=CLIPClassifier)
        classifier._extract_region = CLIPClassifier._extract_region.__get__(classifier)
        
        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test valid region
        roi = classifier._extract_region(image, 10, 10, 50, 50)
        assert roi is not None
        assert roi.shape == (50, 50, 3)
        
        # Test region outside bounds
        roi = classifier._extract_region(image, 90, 90, 50, 50)
        assert roi is not None
        assert roi.shape[0] <= 10  # Should be clipped
        assert roi.shape[1] <= 10
        
        # Test negative coordinates
        roi = classifier._extract_region(image, -5, -5, 20, 20)
        assert roi is not None
        assert roi.shape == (15, 15, 3)  # Should be clipped to valid area
    
    def test_prepare_text_prompts_format(self):
        """Test text prompt formatting."""
        from task1_object_detection.clip_classifier import CLIPClassifier
        
        # Create mock classifier
        classifier = Mock(spec=CLIPClassifier)
        classifier._prepare_text_prompts = CLIPClassifier._prepare_text_prompts.__get__(classifier)
        
        # Mock the clip.tokenize function
        with patch('task1_object_detection.clip_classifier.clip.tokenize') as mock_tokenize:
            mock_tokenize.return_value = Mock()
            mock_tokenize.return_value.to = Mock(return_value="mocked_tensor")
            classifier.device = "cpu"
            
            result = classifier._prepare_text_prompts(["cat", "dog"])
            
            # Verify that prompts were formatted correctly
            expected_calls = mock_tokenize.call_args[0][0]
            assert "a photo of a cat" in expected_calls
            assert "a photo of a dog" in expected_calls
    
    def test_classify_regions_empty_inputs(self):
        """Test classifier behavior with empty inputs."""
        from task1_object_detection.clip_classifier import CLIPClassifier
        
        # Create mock classifier
        classifier = Mock(spec=CLIPClassifier)
        classifier.classify_regions = CLIPClassifier.classify_regions.__get__(classifier)
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test empty regions
        result = classifier.classify_regions(image, [], ["cat", "dog"])
        assert result == []
        
        # Test empty prompts
        result = classifier.classify_regions(image, [(10, 10, 20, 20)], [])
        assert result == []
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        from task1_object_detection.clip_classifier import CLIPClassifier
        
        # Create mock classifier
        classifier = Mock(spec=CLIPClassifier)
        classifier.get_model_info = CLIPClassifier.get_model_info.__get__(classifier)
        classifier.model_name = "ViT-B/32"
        classifier.device = "cpu"
        classifier.confidence_threshold = 0.1
        
        info = classifier.get_model_info()
        
        assert info["model_name"] == "ViT-B/32"
        assert info["device"] == "cpu"
        assert info["confidence_threshold"] == 0.1

def test_benchmark_structure():
    """Test that benchmark function has correct structure."""
    from task1_object_detection.clip_classifier import benchmark_clip_performance
    
    # This is a structural test - we're not actually running CLIP
    # but ensuring the function exists and has the right signature
    import inspect
    sig = inspect.signature(benchmark_clip_performance)
    params = list(sig.parameters.keys())
    
    expected_params = ["image", "regions", "text_prompts"]
    assert all(param in params for param in expected_params)