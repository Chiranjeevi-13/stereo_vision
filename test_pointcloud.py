#!/usr/bin/env python3
"""
Test point cloud generation
"""
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from utils.loader import load_kitti_stereo_pair
from perception.disparity import create_stereo_sgbm, compute_disparity
from perception.depth import compute_depth_map
from perception.detector import ObjectDetector
from perception.localization_3d import localize_objects_3d
from perception.pointcloud import (generate_point_cloud, downsample_point_cloud, 
                                   filter_point_cloud_by_objects, save_point_cloud_ply)
from calibration.parser import load_stereo_params

def main():
    print("="*60)
    print("TESTING POINT CLOUD GENERATION")
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
    stereo = create_stereo_sgbm()
    disparity = compute_disparity(left_img, right_img, stereo)
    depth_map = compute_depth_map(disparity, focal_length, baseline)
    
    # Detect and localize objects
    print("Detecting and localizing objects...")
    detector = ObjectDetector(model_name='yolov8n.pt', confidence=0.5)
    detections = detector.detect(left_img)
    objects_3d = localize_objects_3d(detections, depth_map, stereo_params)
    print(f"  Found {len(objects_3d)} objects")
    
    # Generate point cloud
    print("\nGenerating point cloud...")
    start = time.time()
    points, colors = generate_point_cloud(depth_map, left_img, stereo_params, max_depth=50.0)
    pc_time = time.time() - start
    
    print(f"  Generated {len(points):,} points in {pc_time:.3f}s")
    print(f"  Point cloud bounds:")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] meters")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] meters")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] meters")
    
    # Downsample
    print("\nDownsampling point cloud...")
    start = time.time()
    points_down, colors_down = downsample_point_cloud(points, colors, voxel_size=0.2)
    down_time = time.time() - start
    
    reduction = 100 * (1 - len(points_down) / len(points))
    print(f"  Downsampled to {len(points_down):,} points in {down_time:.3f}s")
    print(f"  Reduction: {reduction:.1f}%")
    
    # Save full point cloud
    print("\nSaving point cloud...")
    save_point_cloud_ply(points_down, colors_down, 'outputs/scene_pointcloud.ply')
    
    # Visualize
    print("\nVisualizing point cloud...")
    print("  (Displaying 10,000 random points for speed)")
    
    # Sample for visualization
    if len(points_down) > 10000:
        indices = np.random.choice(len(points_down), 10000, replace=False)
        vis_points = points_down[indices]
        vis_colors = colors_down[indices]
    else:
        vis_points = points_down
        vis_colors = colors_down
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(vis_points[:, 0], vis_points[:, 2], -vis_points[:, 1],
              c=vis_colors/255.0, s=1, alpha=0.5)
    
    # Plot object positions
    for obj in objects_3d:
        x = obj['position_3d']['X']
        y = obj['position_3d']['Y']
        z = obj['position_3d']['Z']
        ax.scatter([x], [z], [-y], c='red', s=100, marker='o', edgecolors='black')
        ax.text(x, z, -y, f"  {obj['class_name']}", fontsize=10)
    
    ax.set_xlabel('X (meters) - Left/Right')
    ax.set_ylabel('Z (meters) - Forward')
    ax.set_zlabel('Y (meters) - Up/Down')
    ax.set_title(f'3D Point Cloud - Frame {frame_idx}\nRed dots = detected objects')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTest completed successfully!")
    print(f"\nPoint cloud saved to: outputs/scene_pointcloud.ply")
    print("  Open with MeshLab, CloudCompare, or similar software")
    
    return 0

if __name__ == "__main__":
    exit(main())
