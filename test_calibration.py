#!/usr/bin/env python3
"""
Test calibration parsing
"""
from calibration.parser import parse_kitti_calib, extract_stereo_params, save_stereo_params

def main():
    print("="*60)
    print("TESTING KITTI CALIBRATION PARSER")
    print("="*60)
    
    # Path to calibration file
    calib_path = "data/kitti/2011_09_26/calib_cam_to_cam.txt"
    
    print(f"\nParsing: {calib_path}")
    
    # Parse raw calibration
    calib_dict = parse_kitti_calib(calib_path)
    print(f"\nParsed {len(calib_dict)} calibration entries")
    
    # Extract stereo parameters
    stereo_params = extract_stereo_params(calib_dict)
    
    print("\n" + "="*60)
    print("STEREO CAMERA PARAMETERS")
    print("="*60)
    
    print("\nLeft Camera (image_02):")
    print(f"  Focal length: fx={stereo_params['left']['fx']:.2f}, fy={stereo_params['left']['fy']:.2f} pixels")
    print(f"  Principal point: cx={stereo_params['left']['cx']:.2f}, cy={stereo_params['left']['cy']:.2f} pixels")
    print(f"  Distortion coefficients: {stereo_params['left']['D']}")
    
    print("\nRight Camera (image_03):")
    print(f"  Focal length: fx={stereo_params['right']['fx']:.2f}, fy={stereo_params['right']['fy']:.2f} pixels")
    print(f"  Principal point: cx={stereo_params['right']['cx']:.2f}, cy={stereo_params['right']['cy']:.2f} pixels")
    print(f"  Distortion coefficients: {stereo_params['right']['D']}")
    
    print(f"\nBaseline (distance between cameras): {stereo_params['baseline']:.4f} meters")
    print(f"  = {stereo_params['baseline']*100:.2f} cm")
    
    print(f"\nImage size: {stereo_params['image_size'][0]} x {stereo_params['image_size'][1]} pixels")
    
    # Save parameters
    print("\n" + "="*60)
    save_stereo_params(stereo_params)
    
    print("\nCalibration parsing complete")
    print("Parameters saved to: calibration/kitti_stereo_params.yaml")
    
    return 0

if __name__ == "__main__":
    exit(main())
