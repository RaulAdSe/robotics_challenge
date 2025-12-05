"""Code128 barcode decoder implementation without external libraries.

This module decodes Code128 barcodes using:
- Virtual scanline extraction
- Run-length encoding (RLE)
- Pattern matching with the complete Code128 specification
- Checksum verification
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class Code128Decoder:
    """Decode Code128 barcodes using the virtual scanline technique."""

    def __init__(self, num_scanlines: int = 5, tolerance: float = 0.3):
        """
        Initialize Code128 decoder with lookup tables.

        Args:
            num_scanlines: Number of scanlines to try for decoding
            tolerance: Pattern matching tolerance (0.0-1.0)
        """
        self.num_scanlines = num_scanlines
        self.tolerance = tolerance

        # Build complete Code128 pattern tables
        self.code128_patterns = self._build_complete_code128_table()

        # Start codes (value: pattern)
        self.START_A = 103
        self.START_B = 104
        self.START_C = 105
        self.STOP = 106

        # Special function codes
        self.FNC1 = 102
        self.FNC2 = 97
        self.FNC3 = 96
        self.FNC4 = 100  # Code A shift
        self.CODE_A = 101
        self.CODE_B = 100
        self.CODE_C = 99
        self.SHIFT = 98

    def _build_complete_code128_table(self) -> Dict[int, List[int]]:
        """
        Build the complete Code128 pattern lookup table.

        Each pattern is [bar, space, bar, space, bar, space] with widths 1-4.
        Total width is always 11 modules.
        """
        # Complete Code128 specification patterns
        # Format: value -> [B, S, B, S, B, S] where B=bar width, S=space width
        patterns = {
            0: [2, 1, 2, 2, 2, 2],   # Space (Code B: space, Code A: space, Code C: 00)
            1: [2, 2, 2, 1, 2, 2],   # ! / ! / 01
            2: [2, 2, 2, 2, 2, 1],   # " / " / 02
            3: [1, 2, 1, 2, 2, 3],   # # / # / 03
            4: [1, 2, 1, 3, 2, 2],   # $ / $ / 04
            5: [1, 3, 1, 2, 2, 2],   # % / % / 05
            6: [1, 2, 2, 2, 1, 3],   # & / & / 06
            7: [1, 2, 2, 3, 1, 2],   # ' / ' / 07
            8: [1, 3, 2, 2, 1, 2],   # ( / ( / 08
            9: [2, 2, 1, 2, 1, 3],   # ) / ) / 09
            10: [2, 2, 1, 3, 1, 2],  # * / * / 10
            11: [2, 3, 1, 2, 1, 2],  # + / + / 11
            12: [1, 1, 2, 2, 3, 2],  # , / , / 12
            13: [1, 2, 2, 1, 3, 2],  # - / - / 13
            14: [1, 2, 2, 2, 3, 1],  # . / . / 14
            15: [1, 1, 3, 2, 2, 2],  # / / / / 15
            16: [1, 2, 3, 1, 2, 2],  # 0 / 0 / 16
            17: [1, 2, 3, 2, 2, 1],  # 1 / 1 / 17
            18: [2, 2, 3, 2, 1, 1],  # 2 / 2 / 18
            19: [2, 2, 1, 1, 3, 2],  # 3 / 3 / 19
            20: [2, 2, 1, 2, 3, 1],  # 4 / 4 / 20
            21: [2, 1, 3, 2, 1, 2],  # 5 / 5 / 21
            22: [2, 2, 3, 1, 1, 2],  # 6 / 6 / 22
            23: [3, 1, 2, 1, 3, 1],  # 7 / 7 / 23
            24: [3, 1, 1, 2, 2, 2],  # 8 / 8 / 24
            25: [3, 2, 1, 1, 2, 2],  # 9 / 9 / 25
            26: [3, 2, 1, 2, 2, 1],  # : / : / 26
            27: [3, 1, 2, 2, 1, 2],  # ; / ; / 27
            28: [3, 2, 2, 1, 1, 2],  # < / < / 28
            29: [3, 2, 2, 2, 1, 1],  # = / = / 29
            30: [2, 1, 2, 1, 2, 3],  # > / > / 30
            31: [2, 1, 2, 3, 2, 1],  # ? / ? / 31
            32: [2, 3, 2, 1, 2, 1],  # @ / @ / 32
            33: [1, 1, 1, 3, 2, 3],  # A / A / 33
            34: [1, 3, 1, 1, 2, 3],  # B / B / 34
            35: [1, 3, 1, 3, 2, 1],  # C / C / 35
            36: [1, 1, 2, 3, 1, 3],  # D / D / 36
            37: [1, 3, 2, 1, 1, 3],  # E / E / 37
            38: [1, 3, 2, 3, 1, 1],  # F / F / 38
            39: [2, 1, 1, 3, 1, 3],  # G / G / 39
            40: [2, 3, 1, 1, 1, 3],  # H / H / 40
            41: [2, 3, 1, 3, 1, 1],  # I / I / 41
            42: [1, 1, 2, 1, 3, 3],  # J / J / 42
            43: [1, 1, 2, 3, 3, 1],  # K / K / 43
            44: [1, 3, 2, 1, 3, 1],  # L / L / 44
            45: [1, 1, 3, 1, 2, 3],  # M / M / 45
            46: [1, 1, 3, 3, 2, 1],  # N / N / 46
            47: [1, 3, 3, 1, 2, 1],  # O / O / 47
            48: [3, 1, 3, 1, 2, 1],  # P / P / 48
            49: [2, 1, 1, 3, 3, 1],  # Q / Q / 49
            50: [2, 3, 1, 1, 3, 1],  # R / R / 50
            51: [2, 1, 3, 1, 1, 3],  # S / S / 51
            52: [2, 1, 3, 3, 1, 1],  # T / T / 52
            53: [2, 1, 3, 1, 3, 1],  # U / U / 53
            54: [3, 1, 1, 1, 2, 3],  # V / V / 54
            55: [3, 1, 1, 3, 2, 1],  # W / W / 55
            56: [3, 3, 1, 1, 2, 1],  # X / X / 56
            57: [3, 1, 2, 1, 1, 3],  # Y / Y / 57
            58: [3, 1, 2, 3, 1, 1],  # Z / Z / 58
            59: [3, 3, 2, 1, 1, 1],  # [ / [ / 59
            60: [3, 1, 4, 1, 1, 1],  # \ / \ / 60
            61: [2, 2, 1, 4, 1, 1],  # ] / ] / 61
            62: [4, 3, 1, 1, 1, 1],  # ^ / ^ / 62
            63: [1, 1, 1, 2, 2, 4],  # _ / _ / 63
            64: [1, 1, 1, 4, 2, 2],  # NUL / ` / 64
            65: [1, 2, 1, 1, 2, 4],  # SOH / a / 65
            66: [1, 2, 1, 4, 2, 1],  # STX / b / 66
            67: [1, 4, 1, 1, 2, 2],  # ETX / c / 67
            68: [1, 4, 1, 2, 2, 1],  # EOT / d / 68
            69: [1, 1, 2, 2, 1, 4],  # ENQ / e / 69
            70: [1, 1, 2, 4, 1, 2],  # ACK / f / 70
            71: [1, 2, 2, 1, 1, 4],  # BEL / g / 71
            72: [1, 2, 2, 4, 1, 1],  # BS / h / 72
            73: [1, 4, 2, 1, 1, 2],  # HT / i / 73
            74: [1, 4, 2, 2, 1, 1],  # LF / j / 74
            75: [2, 4, 1, 2, 1, 1],  # VT / k / 75
            76: [2, 2, 1, 1, 1, 4],  # FF / l / 76
            77: [4, 1, 3, 1, 1, 1],  # CR / m / 77
            78: [2, 4, 1, 1, 1, 2],  # SO / n / 78
            79: [1, 3, 4, 1, 1, 1],  # SI / o / 79
            80: [1, 1, 1, 2, 4, 2],  # DLE / p / 80
            81: [1, 2, 1, 1, 4, 2],  # DC1 / q / 81
            82: [1, 2, 1, 2, 4, 1],  # DC2 / r / 82
            83: [1, 1, 4, 2, 1, 2],  # DC3 / s / 83
            84: [1, 2, 4, 1, 1, 2],  # DC4 / t / 84
            85: [1, 2, 4, 2, 1, 1],  # NAK / u / 85
            86: [4, 1, 1, 2, 1, 2],  # SYN / v / 86
            87: [4, 2, 1, 1, 1, 2],  # ETB / w / 87
            88: [4, 2, 1, 2, 1, 1],  # CAN / x / 88
            89: [2, 1, 2, 1, 4, 1],  # EM / y / 89
            90: [2, 1, 4, 1, 2, 1],  # SUB / z / 90
            91: [4, 1, 2, 1, 2, 1],  # ESC / { / 91
            92: [1, 1, 1, 1, 4, 3],  # FS / | / 92
            93: [1, 1, 1, 3, 4, 1],  # GS / } / 93
            94: [1, 3, 1, 1, 4, 1],  # RS / ~ / 94
            95: [1, 1, 4, 1, 1, 3],  # US / DEL / 95
            96: [1, 1, 4, 3, 1, 1],  # FNC 3 / FNC 3 / 96
            97: [4, 1, 1, 1, 1, 3],  # FNC 2 / FNC 2 / 97
            98: [4, 1, 1, 3, 1, 1],  # SHIFT B / SHIFT A / 98
            99: [1, 1, 3, 1, 4, 1],  # CODE C / CODE C / 99
            100: [1, 1, 4, 1, 3, 1], # CODE B / FNC 4 / CODE B
            101: [3, 1, 1, 1, 4, 1], # FNC 4 / CODE A / CODE A
            102: [4, 1, 1, 1, 3, 1], # FNC 1 / FNC 1 / FNC 1
            103: [2, 1, 1, 4, 1, 2], # Start A
            104: [2, 1, 1, 2, 1, 4], # Start B
            105: [2, 1, 1, 2, 3, 2], # Start C
            106: [2, 3, 3, 1, 1, 1], # Stop (followed by 2-module bar)
        }
        return patterns

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
            "scanlines_tried": 0,
            "best_scanline": None,
            "run_lengths": [],
            "normalized_patterns": [],
            "detected_characters": [],
            "checksum_valid": False,
            "decoding_stages": {}
        }

        # Convert to grayscale if needed
        if len(barcode_image.shape) == 3:
            gray = cv2.cvtColor(barcode_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = barcode_image.copy()

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Try multiple scanlines
        height, width = gray.shape
        best_result = None
        best_confidence = -1

        # Generate scanline positions
        scanline_positions = self._generate_scanline_positions(height)

        for scanline_y in scanline_positions:
            debug_info["scanlines_tried"] += 1

            # Extract scanline
            scanline = gray[scanline_y, :]

            # Try both directions (left-to-right and right-to-left)
            for direction in ['forward', 'reverse']:
                if direction == 'reverse':
                    scanline = scanline[::-1]

                # Convert to run-length encoding
                run_lengths = self._scanline_to_rle(scanline)

                if len(run_lengths) < 6:
                    continue

                # Try to decode
                result = self._try_decode(run_lengths, debug)

                if result is not None:
                    decoded_text, confidence, decode_debug = result

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = decoded_text
                        debug_info["best_scanline"] = scanline_y
                        debug_info["run_lengths"] = run_lengths
                        debug_info.update(decode_debug)

        if best_result is None:
            debug_info["error"] = "Could not decode barcode from any scanline"
            return None, debug_info

        return best_result, debug_info

    def _generate_scanline_positions(self, height: int) -> List[int]:
        """Generate evenly spaced scanline positions."""
        positions = []
        step = height // (self.num_scanlines + 1)
        for i in range(1, self.num_scanlines + 1):
            y = i * step
            positions.append(y)
            # Also try adjacent lines for better coverage
            if y > 1:
                positions.append(y - 1)
            if y < height - 1:
                positions.append(y + 1)
        return sorted(set(positions))

    def _scanline_to_rle(self, scanline: np.ndarray) -> List[int]:
        """
        Convert scanline to run-length encoding.

        Args:
            scanline: 1D array of pixel intensities

        Returns:
            List of run lengths [bar_width, space_width, bar_width, ...]
        """
        # Use Otsu's thresholding for better binarization
        threshold = self._calculate_threshold(scanline)

        # Convert to binary (bars are dark = 1, spaces are light = 0)
        binary = (scanline < threshold).astype(np.uint8)

        # Find run lengths
        run_lengths = []
        if len(binary) == 0:
            return run_lengths

        # Skip leading quiet zone (white space)
        start_idx = 0
        while start_idx < len(binary) and binary[start_idx] == 0:
            start_idx += 1

        if start_idx >= len(binary):
            return run_lengths

        current_value = binary[start_idx]
        current_length = 1

        for i in range(start_idx + 1, len(binary)):
            if binary[i] == current_value:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_value = binary[i]
                current_length = 1

        # Add final run (if it's a bar, otherwise skip trailing quiet zone)
        if current_value == 1:
            run_lengths.append(current_length)

        return run_lengths

    def _calculate_threshold(self, scanline: np.ndarray) -> int:
        """Calculate optimal threshold using Otsu's method."""
        # Simple approach: use midpoint between min and max
        min_val = int(np.min(scanline))
        max_val = int(np.max(scanline))

        # Histogram-based threshold
        hist, bins = np.histogram(scanline, bins=256, range=(0, 256))

        # Otsu's method
        total = len(scanline)
        current_max = 0
        threshold = (min_val + max_val) // 2
        sum_total = np.sum(np.arange(256) * hist)
        sum_bg = 0
        weight_bg = 0

        for i in range(256):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += i * hist[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance > current_max:
                current_max = variance
                threshold = i

        return threshold

    def _try_decode(self, run_lengths: List[int], debug: bool) -> Optional[Tuple[str, float, Dict]]:
        """
        Try to decode characters from run lengths.

        Returns:
            Tuple of (decoded_text, confidence, debug_info) or None if failed
        """
        decode_debug = {
            "start_position": -1,
            "code_set": None,
            "detected_characters": [],
            "normalized_patterns": [],
            "checksum_valid": False
        }

        # Find start pattern
        start_pos, code_set = self._find_start_pattern(run_lengths)

        if start_pos == -1:
            return None

        decode_debug["start_position"] = start_pos
        decode_debug["code_set"] = code_set

        # Calculate module width from start pattern
        start_pattern_runs = run_lengths[start_pos:start_pos + 6]
        module_width = sum(start_pattern_runs) / 11.0

        # Decode characters
        pos = start_pos + 6  # Skip start pattern
        char_values = []
        patterns = []
        current_code_set = code_set

        while pos + 6 <= len(run_lengths):
            # Check for stop pattern
            if pos + 7 <= len(run_lengths):
                stop_segment = run_lengths[pos:pos + 6]
                if self._match_pattern(stop_segment, self.code128_patterns[106], module_width):
                    break

            # Extract character pattern
            char_runs = run_lengths[pos:pos + 6]

            # Match against all patterns
            matched_value = self._match_character_pattern(char_runs, module_width)

            if matched_value == -1:
                # Try with adjusted module width
                new_module_width = sum(char_runs) / 11.0
                matched_value = self._match_character_pattern(char_runs, new_module_width)

                if matched_value == -1:
                    break

            char_values.append(matched_value)
            patterns.append(char_runs)
            pos += 6

            # Handle code set switches
            if matched_value == 101:  # CODE A
                current_code_set = 'A'
            elif matched_value == 100:  # CODE B
                current_code_set = 'B'
            elif matched_value == 99:  # CODE C
                current_code_set = 'C'

        if not char_values:
            return None

        decode_debug["normalized_patterns"] = patterns

        # Remove checksum character
        checksum_value = char_values[-1] if char_values else -1
        data_values = char_values[:-1]

        # Verify checksum
        is_valid = self._verify_checksum(data_values, checksum_value, code_set)
        decode_debug["checksum_valid"] = is_valid

        # Convert values to characters
        characters = self._values_to_text(data_values, code_set)
        decode_debug["detected_characters"] = characters

        # Calculate confidence
        confidence = 1.0 if is_valid else 0.5
        confidence *= len(characters) / max(len(data_values), 1)

        if not characters:
            return None

        decoded_text = ''.join(characters)
        return decoded_text, confidence, decode_debug

    def _find_start_pattern(self, run_lengths: List[int]) -> Tuple[int, str]:
        """
        Find the start pattern in run lengths.

        Returns:
            Tuple of (start_position, code_set) where code_set is 'A', 'B', or 'C'
        """
        start_patterns = {
            'A': self.code128_patterns[103],
            'B': self.code128_patterns[104],
            'C': self.code128_patterns[105]
        }

        # Search for start pattern in the first portion of the barcode
        max_search = min(len(run_lengths) - 6, 20)

        for start_pos in range(max_search):
            if start_pos + 6 > len(run_lengths):
                break

            segment = run_lengths[start_pos:start_pos + 6]
            module_width = sum(segment) / 11.0

            for code_set, pattern in start_patterns.items():
                if self._match_pattern(segment, pattern, module_width):
                    return start_pos, code_set

        return -1, ''

    def _match_pattern(self, run_lengths: List[int], pattern: List[int], module_width: float) -> bool:
        """
        Check if run lengths match a pattern within tolerance.

        Args:
            run_lengths: Measured run lengths
            pattern: Expected pattern (module counts)
            module_width: Calculated module width in pixels

        Returns:
            True if patterns match within tolerance
        """
        if len(run_lengths) != len(pattern):
            return False

        for run, expected in zip(run_lengths, pattern):
            expected_width = expected * module_width
            error = abs(run - expected_width) / max(expected_width, 1)
            if error > self.tolerance:
                return False

        return True

    def _match_character_pattern(self, run_lengths: List[int], module_width: float) -> int:
        """
        Match run lengths against all Code128 patterns.

        Returns:
            Matched pattern value (0-106) or -1 if no match
        """
        best_match = -1
        best_error = float('inf')

        for value, pattern in self.code128_patterns.items():
            if len(pattern) != len(run_lengths):
                continue

            # Calculate total error
            total_error = 0
            for run, expected in zip(run_lengths, pattern):
                expected_width = expected * module_width
                total_error += abs(run - expected_width) / max(expected_width, 1)

            avg_error = total_error / len(pattern)

            if avg_error < best_error and avg_error < self.tolerance:
                best_error = avg_error
                best_match = value

        return best_match

    def _verify_checksum(self, data_values: List[int], checksum_value: int, start_code_set: str) -> bool:
        """Verify Code128 checksum."""
        if checksum_value < 0 or not data_values:
            return False

        # Start value
        if start_code_set == 'A':
            checksum = 103
        elif start_code_set == 'B':
            checksum = 104
        elif start_code_set == 'C':
            checksum = 105
        else:
            return False

        # Add weighted values
        for i, value in enumerate(data_values):
            weight = i + 1
            checksum += value * weight

        expected_checksum = checksum % 103

        return expected_checksum == checksum_value

    def _values_to_text(self, values: List[int], initial_code_set: str) -> List[str]:
        """Convert Code128 values to text characters."""
        characters = []
        code_set = initial_code_set
        i = 0

        while i < len(values):
            value = values[i]

            # Handle code set switches
            if value == 101:  # CODE A
                code_set = 'A'
                i += 1
                continue
            elif value == 100:  # CODE B (or FNC4 in Code A)
                if code_set != 'A':  # In Code B or C, this means CODE B
                    code_set = 'B'
                i += 1
                continue
            elif value == 99:  # CODE C
                code_set = 'C'
                i += 1
                continue
            elif value == 98:  # SHIFT
                # Temporary shift to other code set
                i += 1
                if i < len(values):
                    next_value = values[i]
                    if code_set == 'A':
                        char = self._value_to_char(next_value, 'B')
                    else:
                        char = self._value_to_char(next_value, 'A')
                    if char:
                        characters.append(char)
                i += 1
                continue
            elif value in (102, 96, 97):  # FNC codes - skip
                i += 1
                continue

            # Regular character
            char = self._value_to_char(value, code_set)
            if char:
                characters.append(char)

            i += 1

        return characters

    def _value_to_char(self, value: int, code_set: str) -> Optional[str]:
        """Convert a single Code128 value to character."""
        if code_set == 'A':
            # Code A: values 0-63 are ASCII 32-95, 64-95 are ASCII 0-31
            if 0 <= value <= 63:
                return chr(value + 32)
            elif 64 <= value <= 95:
                return ''  # Control characters - skip

        elif code_set == 'B':
            # Code B: values 0-95 are ASCII 32-127
            if 0 <= value <= 95:
                return chr(value + 32)

        elif code_set == 'C':
            # Code C: values 0-99 represent two-digit numbers
            if 0 <= value <= 99:
                return f"{value:02d}"

        return None


def test_decoder():
    """Test the decoder on sample images."""
    import os

    decoder = Code128Decoder(num_scanlines=7, tolerance=0.35)

    # Test on rectified barcode images if available
    test_dir = "../../test_barcodes"
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.endswith(('.jpg', '.png')):
                filepath = os.path.join(test_dir, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    result, debug = decoder.decode_barcode(image, debug=True)
                    print(f"{filename}: {result}")
                    if debug.get("error"):
                        print(f"  Error: {debug['error']}")


if __name__ == "__main__":
    test_decoder()
