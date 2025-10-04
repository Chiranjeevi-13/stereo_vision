#!/usr/bin/env python3
"""
Test disparity computation
"""
import cv2
import matplotlib.pyplot as plt
from utils.loader import load_kitti_stereo_pair
from perception.disparity import create_stereo_sgbm, compute_disparity, normalize_disparity_for_display, apply_colormap
import time

def main():
    print("="*60)
    print("TESTING DISPARITY COMPUTATION")
    print("="*60)
    
    # Load stereo pair
    sequence_path = "data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 0
    
    print(f"\nLoading frame {frame_idx}...")
    left_img, right_img = load_kitti_stereo_pair(sequence_path, frame_idx)
    
    # Create stereo matcher
    print("\nCreating StereoSGBM matcher...")
    stereo = create_stereo_sgbm(
        min_disparity=0,
        num_disparities=128,  # Search range (must be divisible by 16)
        block_size=5          # Matching block size
    )
    
    # Compute disparity
    print("Computing disparity (this takes ~2-5 seconds)...")
    start_time = time.time()
    disparity = compute_disparity(left_img, right_img, stereo)
    elapsed = time.time() - start_time
    
    print(f"\nDisparity computed in {elapsed:.2f} seconds")
    print(f"FPS: {1.0/elapsed:.1f}")
    
    # Print statistics
    valid_disp = disparity[disparity > 0]
    print(f"\nDisparity Statistics:")
    print(f"  Shape: {disparity.shape}")
    print(f"  Valid pixels: {len(valid_disp)} / {disparity.size} ({100*len(valid_disp)/disparity.size:.1f}%)")
    print(f"  Min disparity: {valid_disp.min():.2f} pixels")
    print(f"  Max disparity: {valid_disp.max():.2f} pixels")
    print(f"  Mean disparity: {valid_disp.mean():.2f} pixels")
    
    # Normalize and colorize
    disparity_viz = normalize_disparity_for_display(disparity)
    disparity_color = apply_colormap(disparity_viz)
    
    # Display results
    print("\nDisplaying results (close window to continue)...")
    
    # Convert left image from BGR to RGB for matplotlib
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    disparity_rgb = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(left_rgb)
    axes[0].set_title('Left Image')
    axes[0].axis('off')
    
    axes[1].imshow(disparity, cmap='gray')
    axes[1].set_title('Disparity Map (grayscale)')
    axes[1].axis('off')
    
    axes[2].imshow(disparity_rgb)
    axes[2].set_title('Disparity Map (colored)\nBlue=Far, Red=Close')
    axes[2].axis('off')
    
    plt.suptitle(f'KITTI Disparity - Frame {frame_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nTest completed successfully")
    print("\nInterpretation:")
    print("  - Red areas = close objects (large disparity)")
    print("  - Blue areas = far objects (small disparity)")
    print("  - Black areas = invalid/no match")
    
    return 0

if __name__ == "__main__":
    exit(main())
