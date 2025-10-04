#!/usr/bin/env python3
"""
Test depth map generation
"""
import cv2
import matplotlib.pyplot as plt
import time
from utils.loader import load_kitti_stereo_pair
from perception.disparity import create_stereo_sgbm, compute_disparity
from perception.depth import compute_depth_map, normalize_depth_for_display, apply_depth_colormap, compute_depth_statistics
from calibration.parser import load_stereo_params

def main():
    print("="*60)
    print("TESTING DEPTH MAP GENERATION")
    print("="*60)
    
    # Load calibration parameters
    print("\nLoading calibration parameters...")
    stereo_params = load_stereo_params('calibration/kitti_stereo_params.yaml')
    
    focal_length = stereo_params['left']['fx']
    baseline = stereo_params['baseline']
    
    print(f"  Focal length: {focal_length:.2f} pixels")
    print(f"  Baseline: {baseline:.4f} meters ({baseline*100:.2f} cm)")
    
    # Load stereo pair
    sequence_path = "data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 0
    
    print(f"\nLoading frame {frame_idx}...")
    left_img, right_img = load_kitti_stereo_pair(sequence_path, frame_idx)
    
    # Compute disparity
    print("\nComputing disparity...")
    start_time = time.time()
    stereo = create_stereo_sgbm(min_disparity=0, num_disparities=128, block_size=5)
    disparity = compute_disparity(left_img, right_img, stereo)
    disp_time = time.time() - start_time
    print(f"  Disparity computed in {disp_time:.3f} seconds")
    
    # Compute depth
    print("\nComputing depth map...")
    start_time = time.time()
    depth_map = compute_depth_map(disparity, focal_length, baseline, min_depth=0.5, max_depth=50.0)
    depth_time = time.time() - start_time
    print(f"  Depth computed in {depth_time:.3f} seconds")
    
    total_time = disp_time + depth_time
    print(f"\nTotal processing time: {total_time:.3f} seconds")
    print(f"FPS: {1.0/total_time:.1f}")
    
    # Compute statistics
    stats = compute_depth_statistics(depth_map)
    
    print(f"\nDepth Map Statistics:")
    print(f"  Valid pixels: {stats['valid_pixels']} / {stats['total_pixels']} ({stats['valid_percentage']:.1f}%)")
    print(f"  Depth range: {stats['min_depth']:.2f} - {stats['max_depth']:.2f} meters")
    print(f"  Mean depth: {stats['mean_depth']:.2f} meters")
    print(f"  Median depth: {stats['median_depth']:.2f} meters")
    
    # Visualize
    print("\nGenerating visualizations...")
    depth_viz = normalize_depth_for_display(depth_map, max_display_depth=30.0)
    depth_color = apply_depth_colormap(depth_viz)
    
    # Display
    print("\nDisplaying results (close window to continue)...")
    
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Original image
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title('Left Camera Image')
    axes[0, 0].axis('off')
    
    # Top right: Disparity (for reference)
    axes[0, 1].imshow(disparity, cmap='gray', vmin=0, vmax=80)
    axes[0, 1].set_title('Disparity Map (pixels)')
    axes[0, 1].axis('off')
    
    # Bottom left: Depth grayscale
    axes[1, 0].imshow(depth_viz, cmap='gray')
    axes[1, 0].set_title('Depth Map (grayscale)\nWhite=Close, Black=Far')
    axes[1, 0].axis('off')
    
    # Bottom right: Depth colored
    axes[1, 1].imshow(depth_color_rgb)
    axes[1, 1].set_title('Depth Map (colored)\nRed=Close, Blue=Far')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'KITTI Depth Estimation - Frame {frame_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nTest completed successfully")
    print("\nInterpretation:")
    print("  - Road surface: ~5-15 meters")
    print("  - Nearby cars: ~10-25 meters")
    print("  - Buildings/trees: ~20-40 meters")
    
    return 0

if __name__ == "__main__":
    exit(main())
