import numpy as np
import cv2

def get_object_depth(bbox, depth_map, method='median'):
    """
    Get depth value for a detected object.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        depth_map: Depth map (H x W) in meters
        method: 'center', 'median', or 'mean'
        
    Returns:
        depth: Depth in meters, or None if invalid
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Extract depth values within bounding box
    depth_roi = depth_map[y1:y2, x1:x2]
    
    # Get valid depths (non-zero)
    valid_depths = depth_roi[depth_roi > 0]
    
    if len(valid_depths) == 0:
        return None
    
    if method == 'center':
        # Use center point depth
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        depth = depth_map[cy, cx]
        return depth if depth > 0 else None
        
    elif method == 'median':
        # Use median (robust to outliers)
        return float(np.median(valid_depths))
        
    elif method == 'mean':
        # Use mean
        return float(np.mean(valid_depths))
    
    return None


def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Convert 2D pixel + depth to 3D camera coordinates.
    
    Args:
        u: Pixel column (x coordinate)
        v: Pixel row (y coordinate)
        depth: Depth in meters (Z)
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point (image center)
        
    Returns:
        (X, Y, Z): 3D point in camera frame (meters)
    """
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    return (X, Y, Z)


def localize_objects_3d(detections, depth_map, camera_params, depth_method='median'):
    """
    Localize detected objects in 3D space.
    
    Args:
        detections: List of detection dictionaries from YOLO
        depth_map: Depth map (H x W) in meters
        camera_params: Camera calibration parameters
        depth_method: Method for extracting depth from bbox
        
    Returns:
        localized_objects: List of objects with 3D positions
    """
    fx = camera_params['left']['fx']
    fy = camera_params['left']['fy']
    cx = camera_params['left']['cx']
    cy = camera_params['left']['cy']
    
    localized_objects = []
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # Get bounding box center
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Get depth for this object
        depth = get_object_depth(bbox, depth_map, method=depth_method)
        
        if depth is None or depth <= 0:
            # No valid depth, skip this object
            continue
        
        # Convert to 3D coordinates
        X, Y, Z = pixel_to_3d(center_u, center_v, depth, fx, fy, cx, cy)
        
        # Create localized object
        obj = {
            'class_name': det['class_name'],
            'class_id': det['class_id'],
            'confidence': det['confidence'],
            'bbox': bbox,
            'depth': depth,
            'position_3d': {
                'X': X,  # meters, left-right (positive = right)
                'Y': Y,  # meters, up-down (positive = down)
                'Z': Z   # meters, forward (positive = away from camera)
            },
            'distance': np.sqrt(X**2 + Y**2 + Z**2)  # Euclidean distance
        }
        
        localized_objects.append(obj)
    
    return localized_objects


def draw_3d_positions(image, localized_objects):
    """
    Draw 3D position information on image.
    
    Args:
        image: Input image
        localized_objects: List of localized objects
        
    Returns:
        annotated: Image with 3D info drawn
    """
    annotated = image.copy()
    
    for obj in localized_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        class_name = obj['class_name']
        conf = obj['confidence']
        X, Y, Z = obj['position_3d']['X'], obj['position_3d']['Y'], obj['position_3d']['Z']
        dist = obj['distance']
        
        # Text lines
        line1 = f"{class_name} {conf:.2f}"
        line2 = f"Dist: {dist:.1f}m"
        line3 = f"XYZ: ({X:.1f}, {Y:.1f}, {Z:.1f})"
        
        # Draw text background and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        y_offset = y1 - 10
        for line in [line1, line2, line3]:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Background
            cv2.rectangle(annotated, (x1, y_offset - th - 2), (x1 + tw, y_offset + 2), color, -1)
            
            # Text
            cv2.putText(annotated, line, (x1, y_offset), font, font_scale, (0, 0, 0), thickness)
            
            y_offset -= (th + 5)
    
    return annotated
