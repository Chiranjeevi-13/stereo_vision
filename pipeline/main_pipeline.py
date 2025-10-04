"""
Main Stereo Vision Perception Pipeline
"""

import cv2
import numpy as np
import time
from pathlib import Path

from utils.loader import load_kitti_stereo_pair
from perception.disparity import create_stereo_sgbm, compute_disparity
from perception.depth import compute_depth_map, compute_depth_statistics
from perception.detector import ObjectDetector
from perception.localization_3d import localize_objects_3d, draw_3d_positions
from perception.pointcloud import generate_point_cloud, downsample_point_cloud, save_point_cloud_ply
from calibration.parser import load_stereo_params


class StereoVisionPipeline:
    """
    Complete stereo vision perception pipeline.
    """
    
    def __init__(self, config_path='calibration/kitti_stereo_params.yaml', 
                 yolo_model='yolov8n.pt', yolo_confidence=0.5):
        """
        Initialize pipeline with configuration.
        """
        print("Initializing Stereo Vision Pipeline...")
        
        print("  Loading calibration...")
        self.stereo_params = load_stereo_params(config_path)
        self.focal_length = self.stereo_params['left']['fx']
        self.baseline = self.stereo_params['baseline']
        
        print("  Creating stereo matcher...")
        self.stereo = create_stereo_sgbm(
            min_disparity=0,
            num_disparities=128,
            block_size=5
        )
        
        print("  Loading YOLO detector...")
        self.detector = ObjectDetector(model_name=yolo_model, confidence=yolo_confidence)
        
        self.stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0
        }
        
        print("Pipeline initialized successfully!\n")
    
    def process_stereo_pair(self, left_img, right_img, generate_pc=False, save_outputs=False, output_dir='outputs'):
        """
        Process a stereo image pair through the complete pipeline.
        """
        start_time = time.time()
        
        results = {}
        timings = {}
        
        # Step 1: Compute disparity
        t0 = time.time()
        disparity = compute_disparity(left_img, right_img, self.stereo)
        timings['disparity'] = time.time() - t0
        results['disparity'] = disparity
        
        # Step 2: Compute depth
        t0 = time.time()
        depth_map = compute_depth_map(disparity, self.focal_length, self.baseline, 
                                       min_depth=0.5, max_depth=50.0)
        depth_stats = compute_depth_statistics(depth_map)
        timings['depth'] = time.time() - t0
        results['depth_map'] = depth_map
        results['depth_stats'] = depth_stats
        
        # Step 3: Detect objects
        t0 = time.time()
        detections = self.detector.detect(left_img)
        timings['detection'] = time.time() - t0
        results['detections'] = detections
        
        # Step 4: Localize objects in 3D
        t0 = time.time()
        objects_3d = localize_objects_3d(detections, depth_map, self.stereo_params, 
                                         depth_method='median')
        timings['localization'] = time.time() - t0
        results['objects_3d'] = objects_3d
        
        # Step 5: Generate point cloud (optional)
        if generate_pc:
            t0 = time.time()
            points, colors = generate_point_cloud(depth_map, left_img, self.stereo_params, 
                                                   max_depth=50.0)
            points_down, colors_down = downsample_point_cloud(points, colors, target_points=10000)
            timings['pointcloud'] = time.time() - t0
            results['pointcloud'] = {
                'points': points_down,
                'colors': colors_down,
                'num_points': len(points_down)
            }
        
        # Total time
        total_time = time.time() - start_time
        timings['total'] = total_time
        results['timings'] = timings
        results['fps'] = 1.0 / total_time
        
        # Update statistics
        self.stats['total_frames'] += 1
        self.stats['total_time'] += total_time
        self.stats['avg_fps'] = self.stats['total_frames'] / self.stats['total_time']
        
        # Save outputs if requested
        if save_outputs:
            self._save_outputs(results, left_img, output_dir)
        
        return results
    
    def _save_outputs(self, results, left_img, output_dir='outputs'):
        """Save pipeline outputs to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        frame_id = self.stats['total_frames']
        
        # Save annotated image
        annotated = draw_3d_positions(left_img, results['objects_3d'])
        cv2.imwrite(str(output_path / f'frame_{frame_id:04d}_annotated.png'), annotated)
        
        # Save depth map
        depth_viz = (results['depth_map'] / 50.0 * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / f'frame_{frame_id:04d}_depth.png'), depth_viz)
        
        # Save point cloud if available
        if 'pointcloud' in results:
            save_point_cloud_ply(
                results['pointcloud']['points'],
                results['pointcloud']['colors'],
                str(output_path / f'frame_{frame_id:04d}_pointcloud.ply')
            )
    
    def print_results(self, results):
        """Print pipeline results in readable format."""
        print("="*60)
        print("PIPELINE RESULTS")
        print("="*60)
        
        # Timings
        print("\nPerformance:")
        timings = results['timings']
        print(f"  Disparity:     {timings['disparity']*1000:6.1f} ms")
        print(f"  Depth:         {timings['depth']*1000:6.1f} ms")
        print(f"  Detection:     {timings['detection']*1000:6.1f} ms")
        print(f"  Localization:  {timings['localization']*1000:6.1f} ms")
        if 'pointcloud' in timings:
            print(f"  Point Cloud:   {timings['pointcloud']*1000:6.1f} ms")
        print(f"  Total:         {timings['total']*1000:6.1f} ms ({results['fps']:.1f} FPS)")
        
        # Depth statistics
        print("\nDepth Map:")
        stats = results['depth_stats']
        print(f"  Valid pixels:  {stats['valid_percentage']:.1f}%")
        print(f"  Depth range:   {stats['min_depth']:.2f} - {stats['max_depth']:.2f} m")
        print(f"  Mean depth:    {stats['mean_depth']:.2f} m")
        
        # Detections
        print(f"\nDetected Objects: {len(results['objects_3d'])}")
        for i, obj in enumerate(results['objects_3d']):
            print(f"\n  {i+1}. {obj['class_name'].upper()}")
            print(f"     Confidence:  {obj['confidence']:.2f}")
            print(f"     Position:    X={obj['position_3d']['X']:6.2f}m, "
                  f"Y={obj['position_3d']['Y']:6.2f}m, Z={obj['position_3d']['Z']:6.2f}m")
            print(f"     Distance:    {obj['distance']:.2f}m")
        
        # Point cloud
        if 'pointcloud' in results:
            print(f"\nPoint Cloud: {results['pointcloud']['num_points']:,} points")
        
        print("="*60)
