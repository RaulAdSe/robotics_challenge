"""Code128 barcode decoder implementation without external libraries."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class Code128Decoder:
    """Decode Code128 barcodes using the virtual scanline technique."""
    
    def __init__(self):
        """Initialize Code128 decoder with lookup tables."""
        # Code128 character set mapping
        # Each character is represented as [bar, space, bar, space, bar, space] pattern
        # Values represent relative widths (1-4 modules)
        self.code128_patterns = self._build_code128_table()
        self.character_values = self._build_character_values()
        
        # Start codes
        self.start_a_pattern = [2, 1, 1, 4, 1, 2]
        self.start_b_pattern = [2, 1, 1, 2, 1, 4] 
        self.start_c_pattern = [2, 1, 1, 2, 3, 2]
        
        # Stop pattern  
        self.stop_pattern = [2, 3, 3, 1, 1, 1, 2]  # 7 modules
    
    def decode_barcode(self, barcode_image: np.ndarray, debug: bool = False) -> Tuple[Optional[str], Dict]:
        """
        Decode a Code128 barcode from a rectified image.
        
        Args:
            barcode_image: Rectified barcode image (grayscale or BGR)
            debug: Whether to return debug information
            
        Returns:
            Tuple of (decoded_text, debug_info)
            - decoded_text: Decoded string or None if decoding failed
            - debug_info: Dictionary with decoding details
        """
        debug_info = {
            "input_shape": barcode_image.shape,
            "scanline_position": None,
            "run_lengths": [],
            "normalized_patterns": [],
            "detected_characters": [],
            "checksum_valid": False,
            "decoding_stages": {}
        }
        
        # Step 1: Convert to grayscale if needed
        if len(barcode_image.shape) == 3:
            gray = cv2.cvtColor(barcode_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = barcode_image.copy()
        
        # Step 2: Extract scanline from center of image
        scanline, scanline_pos = self._extract_scanline(gray)
        debug_info["scanline_position"] = scanline_pos
        debug_info["decoding_stages"]["scanline"] = scanline if debug else None
        
        # Step 3: Convert scanline to run-length encoding
        run_lengths = self._scanline_to_rle(scanline)
        debug_info["run_lengths"] = run_lengths
        
        if len(run_lengths) < 6:  # Minimum: start pattern (6 elements)
            debug_info["error"] = f"Insufficient run lengths: got {len(run_lengths)}, need at least 6"
            return None, debug_info
        
        # Step 4: Find start pattern
        start_pos, code_set = self._find_start_pattern(run_lengths)
        debug_info["start_position"] = start_pos
        debug_info["code_set"] = code_set
        
        if start_pos == -1:
            debug_info["error"] = "Start pattern not found"
            return None, debug_info
        
        # Step 5: Decode characters
        decoded_chars, checksum_char, patterns = self._decode_characters(
            run_lengths[start_pos + 6:], code_set)
        debug_info["detected_characters"] = decoded_chars
        debug_info["normalized_patterns"] = patterns
        
        if not decoded_chars:
            debug_info["error"] = "No characters decoded"
            return None, debug_info
        
        # Step 6: Verify checksum
        is_valid = self._verify_checksum(decoded_chars, checksum_char, code_set)
        debug_info["checksum_valid"] = is_valid
        debug_info["checksum_character"] = checksum_char
        
        if not is_valid:
            debug_info["error"] = "Invalid checksum"
            # Return partial result for debugging
            return ''.join(decoded_chars), debug_info
        
        # Step 7: Convert to final string
        decoded_text = ''.join(decoded_chars)
        
        return decoded_text, debug_info
    
    def _extract_scanline(self, gray_image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Extract a horizontal scanline from the center of the barcode."""
        height, width = gray_image.shape
        
        # Use multiple scanlines and pick the best one
        scanlines = []
        positions = []
        
        # Try center line and a few lines around it
        center_y = height // 2
        for offset in [-2, -1, 0, 1, 2]:
            y = center_y + offset
            if 0 <= y < height:
                scanline = gray_image[y, :]
                # Measure contrast (difference between min and max)
                contrast = np.max(scanline) - np.min(scanline)
                scanlines.append((scanline, contrast, y))
        
        # Choose scanline with highest contrast
        best_scanline, _, best_y = max(scanlines, key=lambda x: x[1])
        
        return best_scanline, best_y
    
    def _scanline_to_rle(self, scanline: np.ndarray, threshold: Optional[int] = None) -> List[int]:
        """
        Convert scanline to run-length encoding.
        
        Args:
            scanline: 1D array of pixel intensities
            threshold: Binary threshold (auto-calculated if None)
            
        Returns:
            List of run lengths [bar_width, space_width, bar_width, ...]
        """
        if threshold is None:
            # Use simple threshold at middle value
            threshold = (np.min(scanline) + np.max(scanline)) / 2
        
        # Convert to binary (bars are dark, spaces are light)
        binary = (scanline < threshold).astype(np.uint8)
        
        # Find run lengths
        run_lengths = []
        if len(binary) == 0:
            return run_lengths
            
        current_value = binary[0]
        current_length = 1
        
        for i in range(1, len(binary)):
            if binary[i] == current_value:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_value = binary[i]
                current_length = 1
        
        # Add final run
        run_lengths.append(current_length)
        
        # Ensure we start with a bar (black region = 1 in binary)
        if len(run_lengths) > 0 and binary[0] == 0:  # Started with white space
            run_lengths = run_lengths[1:]  # Skip first white run
        
        # Filter out very short runs (noise)
        filtered_runs = [r for r in run_lengths if r >= 1]
        
        return filtered_runs
    
    def _find_start_pattern(self, run_lengths: List[int]) -> Tuple[int, str]:
        """
        Find the start pattern in run lengths.
        
        Returns:
            Tuple of (start_position, code_set) where code_set is 'A', 'B', or 'C'
        """
        start_patterns = {
            'A': self.start_a_pattern,
            'B': self.start_b_pattern,
            'C': self.start_c_pattern
        }
        
        # Try to find start pattern at beginning of run lengths
        for code_set, pattern in start_patterns.items():
            if len(run_lengths) >= len(pattern):
                normalized = self._normalize_pattern(run_lengths[:len(pattern)])
                if self._patterns_match(normalized, pattern):
                    return 0, code_set
        
        # Try to find start pattern after some quiet zone
        for start_pos in range(1, min(len(run_lengths) - 6, 10)):
            for code_set, pattern in start_patterns.items():
                if start_pos + len(pattern) <= len(run_lengths):
                    segment = run_lengths[start_pos:start_pos + len(pattern)]
                    normalized = self._normalize_pattern(segment)
                    if self._patterns_match(normalized, pattern):
                        return start_pos, code_set
        
        return -1, ''
    
    def _normalize_pattern(self, run_lengths: List[int]) -> List[int]:
        """
        Normalize run lengths to Code128 module widths (1-4).
        
        Each Code128 character has exactly 11 modules.
        """
        if not run_lengths or len(run_lengths) != 6:
            return []
        
        total_width = sum(run_lengths)
        target_modules = 11  # Code128 characters are 11 modules wide
        
        # Calculate module width
        module_width = total_width / target_modules
        
        # Normalize each run length
        normalized = []
        for length in run_lengths:
            modules = round(length / module_width)
            modules = max(1, min(4, modules))  # Clamp to 1-4 range
            normalized.append(modules)
        
        # Ensure total is exactly 11
        total_normalized = sum(normalized)
        if total_normalized != target_modules:
            # Adjust the largest element
            diff = target_modules - total_normalized
            max_idx = normalized.index(max(normalized))
            normalized[max_idx] += diff
            normalized[max_idx] = max(1, min(4, normalized[max_idx]))
        
        return normalized
    
    def _patterns_match(self, pattern1: List[int], pattern2: List[int], tolerance: int = 0) -> bool:
        """Check if two patterns match within tolerance."""
        if len(pattern1) != len(pattern2):
            return False
        
        for p1, p2 in zip(pattern1, pattern2):
            if abs(p1 - p2) > tolerance:
                return False
        
        return True
    
    def _decode_characters(self, run_lengths: List[int], code_set: str) -> Tuple[List[str], int, List[List[int]]]:
        """
        Decode characters from run lengths.
        
        Returns:
            Tuple of (decoded_characters, checksum_character, normalized_patterns)
        """
        characters = []
        patterns = []
        pos = 0
        
        while pos + 6 <= len(run_lengths):
            # Extract 6 elements for one character
            char_runs = run_lengths[pos:pos + 6]
            normalized = self._normalize_pattern(char_runs)
            patterns.append(normalized)
            
            # Look up character
            char_value = self._pattern_to_character(normalized)
            if char_value == -1:
                break  # Unknown pattern, stop decoding
            
            # Convert to character based on code set
            char = self._value_to_character(char_value, code_set)
            characters.append(char)
            
            pos += 6
        
        # Last character should be checksum
        checksum_char = characters.pop() if characters else -1
        
        return characters, checksum_char, patterns
    
    def _pattern_to_character(self, pattern: List[int]) -> int:
        """Convert normalized pattern to Code128 character value."""
        for value, code_pattern in self.code128_patterns.items():
            if self._patterns_match(pattern, code_pattern):
                return value
        return -1  # Pattern not found
    
    def _value_to_character(self, value: int, code_set: str) -> str:
        """Convert Code128 value to character based on code set."""
        if code_set in self.character_values and value in self.character_values[code_set]:
            return self.character_values[code_set][value]
        return f"[{value}]"  # Fallback representation
    
    def _verify_checksum(self, characters: List[str], checksum_char: int, code_set: str) -> bool:
        """Verify Code128 checksum."""
        if checksum_char == -1 or not characters:
            return False
        
        # Calculate expected checksum
        checksum = self._get_start_value(code_set)  # Start character value
        
        for i, char in enumerate(characters):
            char_value = self._character_to_value(char, code_set)
            if char_value == -1:
                return False
            checksum += char_value * (i + 1)  # Position weight starts at 1
        
        expected_checksum = checksum % 103
        
        return expected_checksum == checksum_char
    
    def _get_start_value(self, code_set: str) -> int:
        """Get the value of start character for checksum calculation."""
        if code_set == 'A':
            return 103
        elif code_set == 'B':
            return 104
        elif code_set == 'C':
            return 105
        return 0
    
    def _character_to_value(self, char: str, code_set: str) -> int:
        """Convert character back to Code128 value for checksum."""
        if code_set in self.character_values:
            for value, code_char in self.character_values[code_set].items():
                if code_char == char:
                    return value
        return -1
    
    def _build_code128_table(self) -> Dict[int, List[int]]:
        """Build the complete Code128 pattern lookup table."""
        # This is a simplified version - in practice you'd want the full 103-character table
        # Here's a subset for demonstration
        patterns = {
            # Value: [bar, space, bar, space, bar, space] pattern
            0: [2, 1, 2, 2, 2, 2],   # Space character
            1: [2, 2, 2, 1, 2, 2],   # !
            2: [2, 2, 2, 2, 2, 1],   # "
            # ... (you would include all 103 patterns here)
            # Start patterns
            103: [2, 1, 1, 4, 1, 2], # Start A
            104: [2, 1, 1, 2, 1, 4], # Start B  
            105: [2, 1, 1, 2, 3, 2], # Start C
        }
        
        # Add numeric patterns (simplified)
        for i in range(32, 127):  # ASCII printable characters
            if i not in patterns:
                # Generate a dummy pattern for missing characters
                # In practice, you'd use the official Code128 specification
                patterns[i] = [2, 1, 2, 1, 2, 3]  # Placeholder
        
        return patterns
    
    def _build_character_values(self) -> Dict[str, Dict[int, str]]:
        """Build character value mappings for different code sets."""
        char_values = {
            'A': {},  # Code Set A (uppercase + control)
            'B': {},  # Code Set B (uppercase + lowercase)
            'C': {}   # Code Set C (numeric pairs)
        }
        
        # Code Set A: ASCII 00-95 (control chars + uppercase)
        for i in range(96):
            if i < 32:
                char_values['A'][i] = f"[{i:02d}]"  # Control characters
            else:
                char_values['A'][i] = chr(i)
        
        # Code Set B: ASCII 32-127 (printable characters)
        for i in range(96):
            char_values['B'][i] = chr(i + 32)
        
        # Code Set C: 00-99 (numeric pairs)
        for i in range(100):
            char_values['C'][i] = f"{i:02d}"
        
        return char_values

def test_decoder_with_synthetic_barcode():
    """Test the decoder with a synthetic barcode pattern."""
    # Create a simple synthetic barcode for testing
    # This is a simplified test - real barcodes would have precise timing
    
    decoder = Code128Decoder()
    
    # Create a synthetic scanline representing a simple barcode
    # Pattern: quiet zone + start B + "A" + checksum + stop + quiet zone
    synthetic_pattern = []
    
    # Quiet zone
    synthetic_pattern.extend([255] * 10)
    
    # Start B pattern: [2, 1, 1, 2, 1, 4] scaled to pixels
    scale = 3  # 3 pixels per module
    start_pattern = [2, 1, 1, 2, 1, 4]
    for i, width in enumerate(start_pattern):
        value = 0 if i % 2 == 0 else 255  # Alternate black/white
        synthetic_pattern.extend([value] * (width * scale))
    
    # Character "A" pattern (simplified)
    char_pattern = [2, 1, 2, 1, 2, 3]
    for i, width in enumerate(char_pattern):
        value = 0 if i % 2 == 0 else 255
        synthetic_pattern.extend([value] * (width * scale))
    
    # Quiet zone
    synthetic_pattern.extend([255] * 10)
    
    # Convert to numpy array and create 2D image
    scanline = np.array(synthetic_pattern, dtype=np.uint8)
    barcode_image = np.tile(scanline, (20, 1))  # Make it 20 pixels tall
    
    # Test decoding
    result, debug_info = decoder.decode_barcode(barcode_image, debug=True)
    
    print("Synthetic barcode test:")
    print(f"Decoded: {result}")
    print(f"Debug info: {debug_info}")
    
    return result, debug_info

if __name__ == "__main__":
    # Test the decoder
    test_result = test_decoder_with_synthetic_barcode()
    
    print("\nCode128 decoder test completed!")