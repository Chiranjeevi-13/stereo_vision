import cv2
import numpy as np

def compute_depth_map(disparity, focal_length, baseline, min_depth=0.5, max_depth=50.0):
    """
    Convert disparity map to depth map in meters.
    
    Args:
        disparity: Disparity map (H x W) in pixels
        focal_length: Camera focal length in pixels
        baseline: Stereo baseline in meters
        min_depth: Minimum valid depth in meters
        max_depth: Maximum valid depth in meters
        
    Returns:
        depth_map: Depth map (H x W) in meters
    """
    valid_mask = disparity > 0.1
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
    depth_map[depth_map < min_depth] = 0
    depth_map[depth_map > max_depth] = 0
    return depth_map


def normalize_depth_for_display(depth_map, max_display_depth=30.0):
    """
    Normalize depth map to 0-255 range for visualization.
    """
    valid_depth = depth_map[depth_map > 0]
    
    if len(valid_depth) == 0:
        return np.zeros_like(depth_map, dtype=np.uint8)
    
    depth_clipped = np.clip(depth_map, 0, max_display_depth)
    depth_viz = np.zeros_like(depth_map, dtype=np.uint8)
    valid_mask = depth_map > 0
    depth_viz[valid_mask] = (255 - (depth_clipped[valid_mask] / max_display_depth * 255)).astype(np.uint8)
    
    return depth_viz


def apply_depth_colormap(depth_normalized):
    """
    Apply color map to depth for better visualization.
    """
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_color


def compute_depth_statistics(depth_map):
    """
    Compute statistics about the depth map.
    """
    valid_depth = depth_map[depth_map > 0]
    
    if len(valid_depth) == 0:
        return {
            'valid_pixels': 0,
            'total_pixels': depth_map.size,
            'valid_percentage': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'mean_depth': 0.0,
            'median_depth': 0.0
        }
    
    stats = {
        'valid_pixels': len(valid_depth),
        'total_pixels': depth_map.size,
        'valid_percentage': 100.0 * len(valid_depth) / depth_map.size,
        'min_depth': float(valid_depth.min()),
        'max_depth': float(valid_depth.max()),
        'mean_depth': float(valid_depth.mean()),
        'median_depth': float(np.median(valid_depth))
    }
    
    return stats
