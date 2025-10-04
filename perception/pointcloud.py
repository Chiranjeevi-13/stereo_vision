import numpy as np
import cv2

def generate_point_cloud(depth_map, rgb_image, camera_params, max_depth=50.0):
    """
    Generate 3D point cloud from depth map and RGB image.
    
    Args:
        depth_map: Depth map (H x W) in meters
        rgb_image: Color image (H x W x 3) BGR
        camera_params: Camera calibration parameters
        max_depth: Maximum depth to include (meters)
        
    Returns:
        points: Nx3 array of 3D coordinates (X, Y, Z)
        colors: Nx3 array of RGB colors (0-255)
    """
    fx = camera_params['left']['fx']
    fy = camera_params['left']['fy']
    cx = camera_params['left']['cx']
    cy = camera_params['left']['cy']
    
    height, width = depth_map.shape
    
    # Create coordinate grids
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten to 1D arrays
    u = u_coords.flatten()
    v = v_coords.flatten()
    z = depth_map.flatten()
    
    # Filter valid depths
    valid = (z > 0) & (z < max_depth)
    
    u = u[valid]
    v = v[valid]
    z = z[valid]
    
    # Convert to 3D coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into Nx3 array
    points = np.stack([x, y, z], axis=1)
    
    # Get colors (convert BGR to RGB)
    rgb_flat = rgb_image.reshape(-1, 3)
    colors = rgb_flat[valid][:, ::-1]
    
    return points, colors


def downsample_point_cloud(points, colors, target_points=10000):
    """
    Downsample point cloud using random sampling.
    
    This is much faster than voxel grid filtering and produces
    good results for visualization purposes.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        target_points: Target number of points after downsampling
        
    Returns:
        downsampled_points: Mx3 array (M <= target_points)
        downsampled_colors: Mx3 array
    """
    n_points = len(points)
    
    # If we already have fewer points than target, return as-is
    if n_points <= target_points:
        return points, colors
    
    # Random sampling without replacement
    indices = np.random.choice(n_points, target_points, replace=False)
    
    # Select the random subset
    downsampled_points = points[indices]
    downsampled_colors = colors[indices]
    
    return downsampled_points, downsampled_colors


def filter_point_cloud_by_objects(points, colors, localized_objects, margin=0.5):
    """
    Extract point cloud regions around detected objects.
    
    Args:
        points: Nx3 array of all points
        colors: Nx3 array of colors
        localized_objects: List of 3D localized objects
        margin: Margin around object position (meters)
        
    Returns:
        object_clouds: List of (points, colors) for each object
    """
    object_clouds = []
    
    for obj in localized_objects:
        obj_x = obj['position_3d']['X']
        obj_y = obj['position_3d']['Y']
        obj_z = obj['position_3d']['Z']
        
        # Find points within margin of object position
        distances = np.sqrt(
            (points[:, 0] - obj_x)**2 + 
            (points[:, 1] - obj_y)**2 + 
            (points[:, 2] - obj_z)**2
        )
        
        mask = distances < margin
        
        if np.any(mask):
            obj_points = points[mask]
            obj_colors = colors[mask]
            object_clouds.append({
                'class_name': obj['class_name'],
                'points': obj_points,
                'colors': obj_colors,
                'num_points': len(obj_points)
            })
    
    return object_clouds


def save_point_cloud_ply(points, colors, filename):
    """
    Save point cloud to PLY file format.
    
    Args:
        points: Nx3 array of points
        colors: Nx3 array of RGB colors (0-255)
        filename: Output filename (e.g., 'cloud.ply')
    """
    n_points = len(points)
    
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"Saved point cloud to {filename} ({n_points} points)")
