#!/usr/bin/env python3
"""
Demo of complete stereo vision pipeline
"""
import cv2
import matplotlib.pyplot as plt
from utils.loader import load_kitti_stereo_pair
from pipeline.main_pipeline import StereoVisionPipeline

def main():
    print("="*60)
    print("STEREO VISION PIPELINE DEMO")
    print("="*60)
    print()
    
    # Initialize pipeline
    pipeline = StereoVisionPipeline(
        config_path='calibration/kitti_stereo_params.yaml',
        yolo_model='yolov8n.pt',
        yolo_confidence=0.5
    )
    
    # Load test image
    sequence_path = "data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 0
    
    print(f"Loading frame {frame_idx}...")
    left_img, right_img = load_kitti_stereo_pair(sequence_path, frame_idx)
    print()
    
    # Process through pipeline
    print("Processing through pipeline...")
    results = pipeline.process_stereo_pair(
        left_img, right_img,
        generate_pc=True,
        save_outputs=True,
        output_dir='outputs'
    )
    print()
    
    # Print results
    pipeline.print_results(results)
    
    # Visualize
    print("\nGenerating visualization...")
    from perception.localization_3d import draw_3d_positions
    annotated = draw_3d_positions(left_img, results['objects_3d'])
    
    # Display
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    depth_viz = (results['depth_map'] / 50.0 * 255).astype('uint8')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(left_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(depth_viz, cmap='jet')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    axes[2].imshow(annotated_rgb)
    axes[2].set_title(f'3D Localized Objects ({len(results["objects_3d"])} found)')
    axes[2].axis('off')
    
    plt.suptitle('Stereo Vision Pipeline Output', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nDemo completed!")
    print(f"Outputs saved to: outputs/")
    
    return 0

if __name__ == "__main__":
    exit(main())
