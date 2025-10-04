import cv2
import numpy as np

def create_stereo_sgbm(min_disparity=0, num_disparities=128, block_size=5):
    """
    Create StereoSGBM object with tuned parameters.
    
    Args:
        min_disparity: Minimum disparity (usually 0)
        num_disparities: Maximum disparity minus minimum (must be divisible by 16)
        block_size: Matched block size (odd number, 3-11 typical)
        
    Returns:
        stereo: StereoSGBM object
    """
    # Calculate P1 and P2 based on block size
    # P1: penalty for small disparity changes (smoothness)
    # P2: penalty for large disparity changes (stronger smoothness)
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,        # Max allowed difference (in integer pixel units) in left-right consistency check
        uniquenessRatio=10,     # Margin in percentage by which best computed cost should "win" second best
        speckleWindowSize=100,  # Maximum size of smooth disparity regions (0 to disable)
        speckleRange=32,        # Maximum disparity variation within each connected component
        preFilterCap=63,        # Truncation value for prefiltered image pixels
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # More accurate but slower mode
    )
    
    return stereo


def compute_disparity(left_img, right_img, stereo=None):
    """
    Compute disparity map from stereo pair.
    
    Args:
        left_img: Left image (H x W x 3) BGR
        right_img: Right image (H x W x 3) BGR
        stereo: StereoSGBM object (if None, creates default)
        
    Returns:
        disparity: Disparity map (H x W) in pixels (float32)
    """
    # Convert to grayscale (SGBM works on grayscale)
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Create stereo matcher if not provided
    if stereo is None:
        stereo = create_stereo_sgbm()
    
    # Compute disparity
    # Result is in fixed-point format (16-bit signed) with 4 fractional bits
    disparity_fixed = stereo.compute(left_gray, right_gray)
    
    # Convert from fixed-point to floating-point
    disparity = disparity_fixed.astype(np.float32) / 16.0
    
    # Invalid disparities are marked as negative or very large
    # Set them to 0 for visualization
    disparity[disparity < 0] = 0
    
    return disparity


def normalize_disparity_for_display(disparity):
    """
    Normalize disparity map to 0-255 range for visualization.
    
    Args:
        disparity: Disparity map (H x W) float
        
    Returns:
        disparity_viz: Normalized disparity (H x W) uint8
    """
    # Get valid (non-zero) disparities
    valid_disp = disparity[disparity > 0]
    
    if len(valid_disp) == 0:
        return np.zeros_like(disparity, dtype=np.uint8)
    
    # Normalize to 0-255 based on min/max of valid disparities
    min_disp = valid_disp.min()
    max_disp = valid_disp.max()
    
    disparity_viz = np.zeros_like(disparity, dtype=np.uint8)
    valid_mask = disparity > 0
    disparity_viz[valid_mask] = ((disparity[valid_mask] - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)
    
    return disparity_viz


def apply_colormap(disparity_normalized):
    """
    Apply color map to disparity for better visualization.
    
    Args:
        disparity_normalized: Normalized disparity (H x W) uint8
        
    Returns:
        disparity_color: Colored disparity map (H x W x 3) BGR
    """
    # Apply TURBO colormap: blue (far) to red (close)
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_TURBO)
    
    return disparity_color
