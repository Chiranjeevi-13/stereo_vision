import numpy as np
import yaml
import os

def parse_kitti_calib(calib_file_path):
    """
    Parse KITTI calibration file (calib_cam_to_cam.txt).
    
    Args:
        calib_file_path: Path to calib_cam_to_cam.txt
        
    Returns:
        calib_dict: Dictionary with calibration parameters
    """
    calib = {}
    
    with open(calib_file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip() == '':
                continue
            
            # Split line into key and values
            parts = line.strip().split(' ')
            key = parts[0].rstrip(':')  # Remove colon from key
            values = parts[1:]
            
            # Try to convert to floats, if it fails keep as string
            try:
                if len(values) > 1:
                    # Multiple values - try to make array
                    calib[key] = np.array([float(v) for v in values])
                elif len(values) == 1:
                    # Single value - try to make float, otherwise keep as string
                    try:
                        calib[key] = float(values[0])
                    except ValueError:
                        calib[key] = values[0]
            except ValueError:
                # If conversion fails, store as string
                calib[key] = ' '.join(values)
    
    return calib


def extract_stereo_params(calib_dict):
    """
    Extract stereo camera parameters for cameras 02 and 03.
    
    Args:
        calib_dict: Raw calibration dictionary from parse_kitti_calib
        
    Returns:
        stereo_params: Dictionary with organized stereo parameters
    """
    # Camera 02 (left color camera)
    K_02 = calib_dict['K_02'].reshape(3, 3)
    D_02 = calib_dict['D_02']
    R_rect_02 = calib_dict['R_rect_02'].reshape(3, 3)
    P_rect_02 = calib_dict['P_rect_02'].reshape(3, 4)
    
    # Camera 03 (right color camera)
    K_03 = calib_dict['K_03'].reshape(3, 3)
    D_03 = calib_dict['D_03']
    R_rect_03 = calib_dict['R_rect_03'].reshape(3, 3)
    P_rect_03 = calib_dict['P_rect_03'].reshape(3, 4)
    
    # Extract focal length and principal point from camera matrices
    fx_left = K_02[0, 0]
    fy_left = K_02[1, 1]
    cx_left = K_02[0, 2]
    cy_left = K_02[1, 2]
    
    fx_right = K_03[0, 0]
    fy_right = K_03[1, 1]
    cx_right = K_03[0, 2]
    cy_right = K_03[1, 2]
    
    # Calculate baseline from projection matrices
    # P_rect_02[0, 3] is approximately 0 (left camera is reference)
    # P_rect_03[0, 3] = -fx * baseline
    baseline = -P_rect_03[0, 3] / P_rect_02[0, 0]
    
    stereo_params = {
        'left': {
            'K': K_02,
            'D': D_02,
            'R_rect': R_rect_02,
            'P_rect': P_rect_02,
            'fx': fx_left,
            'fy': fy_left,
            'cx': cx_left,
            'cy': cy_left
        },
        'right': {
            'K': K_03,
            'D': D_03,
            'R_rect': R_rect_03,
            'P_rect': P_rect_03,
            'fx': fx_right,
            'fy': fy_right,
            'cx': cx_right,
            'cy': cy_right
        },
        'baseline': baseline,
        'image_size': [1242, 375]  # KITTI color images are 1242x375
    }
    
    return stereo_params


def save_stereo_params(stereo_params, output_path='calibration/kitti_stereo_params.yaml'):
    """
    Save stereo parameters to YAML file.
    
    Args:
        stereo_params: Dictionary from extract_stereo_params
        output_path: Where to save the YAML file
    """
    # Convert numpy arrays to lists for YAML serialization
    save_dict = {}
    
    for camera in ['left', 'right']:
        save_dict[camera] = {
            'K': stereo_params[camera]['K'].tolist(),
            'D': stereo_params[camera]['D'].tolist(),
            'R_rect': stereo_params[camera]['R_rect'].tolist(),
            'P_rect': stereo_params[camera]['P_rect'].tolist(),
            'fx': float(stereo_params[camera]['fx']),
            'fy': float(stereo_params[camera]['fy']),
            'cx': float(stereo_params[camera]['cx']),
            'cy': float(stereo_params[camera]['cy'])
        }
    
    save_dict['baseline'] = float(stereo_params['baseline'])
    save_dict['image_size'] = stereo_params['image_size']
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to YAML
    with open(output_path, 'w') as f:
        yaml.dump(save_dict, f, default_flow_style=False)
    
    print(f"Saved stereo parameters to {output_path}")


def load_stereo_params(param_path='calibration/kitti_stereo_params.yaml'):
    """
    Load stereo parameters from YAML file.
    
    Args:
        param_path: Path to saved parameters
        
    Returns:
        stereo_params: Dictionary with numpy arrays
    """
    with open(param_path, 'r') as f:
        save_dict = yaml.safe_load(f)
    
    # Convert lists back to numpy arrays
    stereo_params = {}
    
    for camera in ['left', 'right']:
        stereo_params[camera] = {
            'K': np.array(save_dict[camera]['K']),
            'D': np.array(save_dict[camera]['D']),
            'R_rect': np.array(save_dict[camera]['R_rect']),
            'P_rect': np.array(save_dict[camera]['P_rect']),
            'fx': save_dict[camera]['fx'],
            'fy': save_dict[camera]['fy'],
            'cx': save_dict[camera]['cx'],
            'cy': save_dict[camera]['cy']
        }
    
    stereo_params['baseline'] = save_dict['baseline']
    stereo_params['image_size'] = save_dict['image_size']
    
    return stereo_params
