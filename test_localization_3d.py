#!/usr/bin/env python3
"""
Test 3D object localization
"""
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from utils.loader import load_kitti_stereo_pair
from perception.disparity import create_stereo_sgbm, compute_disparity
from perception.depth import compute_depth_map
from perception.detector import ObjectDetector
from perception.localization_3d import localize_objects_3d, draw_3d_positions
from calibration.parser import load_stereo_params

def main():
    print("="*60)
    print("TESTING 3D OBJECT LOCALIZATION")
    print("="*60)
    
    # Load calibration
    print("\nLoading calibration...")
    stereo_params = load_stereo_params('calibration/kitti_stereo_params.yaml')
    focal_length = stereo_params['left']['fx']
    baseline = stereo_params['baseline']
    
    # Load stereo pair
    sequence_path = "data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 0
    
    print(f"Loading frame {frame_idx}...")
    left_img, right_img = load_kitti_stereo_pair(sequence_path, frame_idx)
    
    # Compute depth
    print("\nComputing depth map...")
    start = time.time()
    stereo = create_stereo_sgbm()
    disparity = compute_disparity(left_img, right_img, stereo)
    depth_map = compute_depth_map(disparity, focal_length, baseline)
    depth_time = time.time() - start
    print(f"  Depth computed in {depth_time:.3f}s")
    
    # Detect objects
    print("\nDetecting objects...")
    start = time.time()
    detector = ObjectDetector(model_name='yolov8n.pt', confidence=0.5)
    detections = detector.detect(left_img)
    detect_time = time.time() - start
    print(f"  Detected {len(detections)} objects in {detect_time:.3f}s")
    
    # Localize in 3D
    print("\nLocalizing objects in 3D...")
    start = time.time()
    objects_3d = localize_objects_3d(detections, depth_map, stereo_params, depth_method='median')
    loc_time = time.time() - start
    print(f"  Localized {len(objects_3d)} objects in {loc_time:.3f}s")
    
    total_time = depth_time + detect_time + loc_time
    print(f"\nTotal pipeline time: {total_time:.3f}s ({1/total_time:.1f} FPS)")
    
    # Print results
    print("\n" + "="*60)
    print("3D LOCALIZED OBJECTS")
    print("="*60)
    
    for i, obj in enumerate(objects_3d):
        print(f"\n{i+1}. {obj['class_name'].upper()}")
        print(f"   Confidence: {obj['confidence']:.2f}")
        print(f"   Depth: {obj['depth']:.2f} meters")
        print(f"   3D Position (camera frame):")
        print(f"     X: {obj['position_3d']['X']:6.2f} m (left/right)")
        print(f"     Y: {obj['position_3d']['Y']:6.2f} m (up/down)")
        print(f"     Z: {obj['position_3d']['Z']:6.2f} m (forward)")
        print(f"   Distance: {obj['distance']:.2f} meters")
    
    # Visualize
    print("\nGenerating visualization...")
    annotated = draw_3d_positions(left_img, objects_3d)
    
    # Display
    print("\nDisplaying results (close window to continue)...")
    
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(left_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f'3D Localized Objects ({len(objects_3d)} objects)')
    axes[1].axis('off')
    
    plt.suptitle(f'3D Object Localization - Frame {frame_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nTest completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())
