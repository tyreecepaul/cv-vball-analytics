import numpy as np
import cv2

def compute_homography(src_points, dst_points):
    """Compute perspective transform matrix."""
    H, _ = cv2.findHomography(np.array(src_points), np.array(dst_points))
    return H

def apply_homography(point, H):
    """Transform a single point."""
    p = np.array([point[0], point[1], 1.0])
    p_transformed = H @ p
    p_transformed /= p_transformed[2]
    return (p_transformed[0], p_transformed[1])
