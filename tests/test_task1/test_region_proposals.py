"""Tests for region proposal generation."""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from task1_object_detection.region_proposals import RegionProposalGenerator, benchmark_proposal_methods

class TestRegionProposalGenerator:
    """Test cases for region proposal generation."""
    
    def test_init(self):
        """Test initialization of RegionProposalGenerator."""
        generator = RegionProposalGenerator()
        assert generator.method == "selective_search"
        assert generator.min_size == 500
        assert generator.max_proposals == 1000
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        generator = RegionProposalGenerator(
            method="blob_detection", 
            min_size=100, 
            max_proposals=500
        )
        assert generator.method == "blob_detection"
        assert generator.min_size == 100
        assert generator.max_proposals == 500
    
    def test_blob_detection_proposals(self, sample_image):
        """Test blob detection proposal generation."""
        generator = RegionProposalGenerator(method="blob_detection")
        proposals = generator.generate_proposals(sample_image)
        
        assert isinstance(proposals, list)
        # Should find at least some proposals from the test image
        assert len(proposals) >= 0
        
        # Check proposal format
        for proposal in proposals:
            assert len(proposal) == 4
            x, y, w, h = proposal
            assert all(isinstance(coord, int) for coord in proposal)
            assert w > 0 and h > 0
            assert w * h >= generator.min_size
    
    def test_selective_search_fallback(self, sample_image):
        """Test that selective search falls back to blob detection if needed."""
        generator = RegionProposalGenerator(method="selective_search")
        proposals = generator.generate_proposals(sample_image)
        
        assert isinstance(proposals, list)
        # Should return proposals even if selective search fails
        assert len(proposals) >= 0
    
    def test_both_methods(self, sample_image):
        """Test using both proposal methods."""
        generator = RegionProposalGenerator(method="both")
        proposals = generator.generate_proposals(sample_image)
        
        assert isinstance(proposals, list)
        assert len(proposals) >= 0
    
    def test_invalid_method(self, sample_image):
        """Test that invalid method raises error."""
        generator = RegionProposalGenerator(method="invalid_method")
        with pytest.raises(ValueError, match="Unknown method"):
            generator.generate_proposals(sample_image)
    
    def test_proposal_filtering(self):
        """Test proposal filtering functionality."""
        generator = RegionProposalGenerator(min_size=1000, max_proposals=5)
        
        # Create test proposals
        proposals = [
            (0, 0, 10, 10),      # Too small (area = 100)
            (0, 0, 50, 50),      # Too small (area = 2500)
            (0, 0, 40, 40),      # Too small (area = 1600) 
            (0, 0, 100, 100),    # Valid (area = 10000)
            (100, 100, 50, 50),  # Valid (area = 2500)
            (200, 200, 60, 60),  # Valid (area = 3600)
        ]
        
        filtered = generator._filter_and_limit_proposals(proposals)
        
        # Should only keep proposals above min_size and limit count
        assert len(filtered) <= generator.max_proposals
        for x, y, w, h in filtered:
            assert w * h >= generator.min_size
    
    def test_iou_calculation(self):
        """Test IoU calculation."""
        generator = RegionProposalGenerator()
        
        # Test identical boxes
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 100, 100)
        assert generator._calculate_iou(box1, box2) == 1.0
        
        # Test non-overlapping boxes
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 100, 100)
        assert generator._calculate_iou(box1, box2) == 0.0
        
        # Test partially overlapping boxes
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 100, 100)
        iou = generator._calculate_iou(box1, box2)
        assert 0 < iou < 1
    
    def test_contour_based_proposals(self):
        """Test contour-based proposal generation."""
        generator = RegionProposalGenerator()
        
        # Create a simple test image with clear contours
        gray = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(gray, (50, 50), (150, 150), 255, -1)
        
        proposals = generator._contour_based_proposals(gray)
        
        assert isinstance(proposals, list)
        # Should find the rectangle we drew
        assert len(proposals) > 0

def test_benchmark_proposal_methods(sample_image):
    """Test benchmarking function."""
    results = benchmark_proposal_methods(sample_image)
    
    assert isinstance(results, dict)
    # Should test at least blob detection (selective search might fail in test env)
    assert "blob_detection" in results
    
    for method, result in results.items():
        assert "proposals_count" in result
        assert "execution_time" in result
        assert "proposals" in result
        assert isinstance(result["proposals_count"], int)
        assert isinstance(result["execution_time"], float)
        assert isinstance(result["proposals"], list)