"""3D pose estimation for barcode surface normal calculation using PnP."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import math

class BarcodeNormalEstimator:
    """Estimate 3D surface normal of barcode using Perspective-n-Point solving."""
    
    def __init__(self, 
                 barcode_real_size: Tuple[float, float] = (50.0, 15.0),
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize 3D pose estimator.
        
        Args:
            barcode_real_size: (width, height) of barcode in real-world units (mm)
            camera_matrix: 3x3 camera intrinsic matrix (estimated if None)
            dist_coeffs: Distortion coefficients (assumed zero if None)
        """
        self.barcode_real_size = barcode_real_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        # Define 3D model points for barcode (flat rectangle in world coordinates)
        self.object_points = self._create_barcode_3d_model()
    
    def estimate_normal_vector(self, 
                             image: np.ndarray,
                             corners: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Estimate the surface normal vector of a barcode.
        
        Args:
            image: Input image containing the barcode
            corners: 4 corner points of detected barcode in image coordinates
            
        Returns:
            Tuple of (normal_vector, debug_info)
            - normal_vector: 3D unit vector representing surface normal (or None if failed)
            - debug_info: Dictionary with pose estimation details
        """
        debug_info = {
            "input_corners": corners.tolist(),
            "camera_matrix": None,
            "object_points": self.object_points.tolist(),
            "success": False,
            "rvec": None,
            "tvec": None,
            "reprojection_error": None
        }
        
        # Step 1: Ensure camera matrix is available
        if self.camera_matrix is None:
            self.camera_matrix = self._estimate_camera_matrix(image.shape)
        
        debug_info["camera_matrix"] = self.camera_matrix.tolist()
        
        # Step 2: Order corners consistently
        ordered_corners = self._order_corners(corners)
        
        # Step 3: Solve PnP problem
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                ordered_corners.astype(np.float32),
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                debug_info["error"] = "PnP solving failed"
                return None, debug_info
            
            debug_info["success"] = True
            debug_info["rvec"] = rvec.flatten().tolist()
            debug_info["tvec"] = tvec.flatten().tolist()
            
            # Step 4: Calculate reprojection error for validation
            reprojected, _ = cv2.projectPoints(
                self.object_points, rvec, tvec, 
                self.camera_matrix, self.dist_coeffs
            )
            
            reprojection_error = self._calculate_reprojection_error(
                ordered_corners, reprojected.squeeze())
            debug_info["reprojection_error"] = reprojection_error
            
            # Step 5: Calculate surface normal vector
            normal_vector = self._calculate_normal_from_pose(rvec)
            
            return normal_vector, debug_info
            
        except Exception as e:
            debug_info["error"] = f"PnP estimation failed: {str(e)}"
            return None, debug_info
    
    def visualize_normal_vector(self, 
                              image: np.ndarray,
                              corners: np.ndarray,
                              normal_vector: np.ndarray,
                              scale: float = 50.0) -> np.ndarray:
        """
        Visualize the estimated normal vector on the image.
        
        Args:
            image: Input image
            corners: 4 corner points of barcode
            normal_vector: 3D normal vector
            scale: Scale factor for arrow length
            
        Returns:
            Image with normal vector arrow drawn
        """
        result = image.copy()
        
        # Calculate barcode center
        center = np.mean(corners, axis=0).astype(int)
        
        # Project normal vector to 2D image coordinates
        if self.camera_matrix is None:
            self.camera_matrix = self._estimate_camera_matrix(image.shape)
        
        # Create a 3D point in the direction of the normal
        normal_3d = normal_vector * scale
        
        # For visualization, assume the barcode is at origin with the normal pointing outward
        origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        normal_point_3d = np.array([normal_3d], dtype=np.float32)
        
        try:
            # Get pose for projection
            _, debug_info = self.estimate_normal_vector(image, corners)
            
            if debug_info["success"]:
                rvec = np.array(debug_info["rvec"]).reshape(3, 1)
                tvec = np.array(debug_info["tvec"]).reshape(3, 1)
                
                # Project origin and normal endpoint
                origin_proj, _ = cv2.projectPoints(
                    origin_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                normal_proj, _ = cv2.projectPoints(
                    normal_point_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                
                # Draw arrow from center to projected normal
                start_point = tuple(center)
                direction = normal_proj.squeeze() - origin_proj.squeeze()
                
                # Normalize and scale direction vector
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction) * scale
                    end_point = tuple((center + direction).astype(int))
                    
                    # Draw arrow
                    from ..utils.visualization import draw_arrow_3d
                    result = draw_arrow_3d(result, start_point, direction, 
                                         length=int(scale), color=(0, 0, 255), thickness=3)
                    
                    # Add text label
                    cv2.putText(result, "Normal", 
                              (center[0] + 10, center[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Visualization failed: {e}")
            # Fallback: draw simple arrow from center
            end_point = (center[0], center[1] - int(scale))
            cv2.arrowedLine(result, tuple(center), end_point, (0, 0, 255), 3)
        
        return result
    
    def _create_barcode_3d_model(self) -> np.ndarray:
        """
        Create 3D model points for the barcode.
        
        Returns:
            4x3 array of 3D points representing barcode corners
        """
        width, height = self.barcode_real_size
        
        # Define barcode as rectangle in XY plane (Z=0)
        # Order: top-left, top-right, bottom-right, bottom-left
        object_points = np.array([
            [-width/2, -height/2, 0.0],  # Top-left
            [width/2, -height/2, 0.0],   # Top-right  
            [width/2, height/2, 0.0],    # Bottom-right
            [-width/2, height/2, 0.0]    # Bottom-left
        ], dtype=np.float32)
        
        return object_points
    
    def _estimate_camera_matrix(self, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Estimate camera intrinsic matrix from image dimensions.
        
        Args:
            image_shape: (height, width, channels) of input image
            
        Returns:
            3x3 camera matrix
        """
        height, width = image_shape[:2]
        
        # Assume reasonable focal length (about 0.8 * width)
        focal_length = 0.8 * max(width, height)
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return camera_matrix
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered corner points
        """
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center to each corner
        angles = []
        for corner in corners:
            angle = math.atan2(corner[1] - center[1], corner[0] - center[0])
            # Normalize angle to [0, 2π]
            angle = angle if angle >= 0 else angle + 2 * math.pi
            angles.append(angle)
        
        # Sort corners by angle (starting from top-left, going clockwise)
        sorted_indices = np.argsort(angles)
        
        # Reorder corners: we want top-left, top-right, bottom-right, bottom-left
        # Adjust based on which quadrant the first sorted corner is in
        first_corner = corners[sorted_indices[0]]
        
        if first_corner[0] < center[0] and first_corner[1] < center[1]:
            # First corner is top-left, perfect
            ordered_indices = sorted_indices
        elif first_corner[0] > center[0] and first_corner[1] < center[1]:
            # First corner is top-right, shift by 1
            ordered_indices = np.roll(sorted_indices, -1)
        elif first_corner[0] > center[0] and first_corner[1] > center[1]:
            # First corner is bottom-right, shift by 2
            ordered_indices = np.roll(sorted_indices, -2)
        else:
            # First corner is bottom-left, shift by 3
            ordered_indices = np.roll(sorted_indices, -3)
        
        ordered_corners = corners[ordered_indices]
        
        return ordered_corners
    
    def _calculate_reprojection_error(self, 
                                    original_points: np.ndarray,
                                    reprojected_points: np.ndarray) -> float:
        """Calculate RMS reprojection error."""
        if len(original_points) != len(reprojected_points):
            return float('inf')
        
        errors = []
        for orig, reproj in zip(original_points, reprojected_points):
            error = np.linalg.norm(orig - reproj)
            errors.append(error)
        
        rms_error = np.sqrt(np.mean(np.array(errors) ** 2))
        return float(rms_error)
    
    def _calculate_normal_from_pose(self, rvec: np.ndarray) -> np.ndarray:
        """
        Calculate surface normal vector from rotation vector.
        
        Args:
            rvec: 3x1 rotation vector from solvePnP
            
        Returns:
            3D unit normal vector
        """
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # The surface normal in object coordinates is [0, 0, 1] (Z-axis)
        # Transform this to camera coordinates
        object_normal = np.array([0.0, 0.0, 1.0])
        camera_normal = rotation_matrix @ object_normal
        
        # Normalize to unit vector
        normal_length = np.linalg.norm(camera_normal)
        if normal_length > 0:
            camera_normal = camera_normal / normal_length
        
        return camera_normal

def demo_pose_estimation(image_path: str, corners: np.ndarray) -> None:
    """
    Demonstrate pose estimation on a test case.
    
    Args:
        image_path: Path to test image
        corners: 4 corner points of detected barcode
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    estimator = BarcodeNormalEstimator()
    
    # Estimate normal vector
    normal_vector, debug_info = estimator.estimate_normal_vector(image, corners)
    
    print(f"Pose Estimation Results:")
    print(f"Success: {debug_info['success']}")
    
    if normal_vector is not None:
        print(f"Normal vector: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")
        print(f"Reprojection error: {debug_info['reprojection_error']:.2f} pixels")
        
        # Visualize result
        result_image = estimator.visualize_normal_vector(image, corners, normal_vector)
        
        output_path = "demo_pose_estimation.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Visualization saved: {output_path}")
    else:
        print(f"Pose estimation failed: {debug_info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Test pose estimation with synthetic corners
    print("Testing 3D pose estimation...")
    
    # Create test corners (representing a slightly rotated rectangle)
    test_corners = np.array([
        [100, 50],   # Top-left
        [300, 60],   # Top-right
        [290, 150],  # Bottom-right
        [110, 140]   # Bottom-left
    ])
    
    # Test with a dummy image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128
    
    estimator = BarcodeNormalEstimator()
    normal_vector, debug_info = estimator.estimate_normal_vector(test_image, test_corners)
    
    print(f"Test Results:")
    print(f"Success: {debug_info['success']}")
    if normal_vector is not None:
        print(f"Normal vector: {normal_vector}")
        print(f"Reprojection error: {debug_info['reprojection_error']:.2f}")
    else:
        print(f"Error: {debug_info.get('error', 'Unknown')}")
    
    print("3D pose estimation test completed!")