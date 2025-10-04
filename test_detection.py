#!/usr/bin/env python3
"""
Test YOLO object detection
"""
import cv2
import matplotlib.pyplot as plt
import time
from utils.loader import load_kitti_stereo_pair
from perception.detector import ObjectDetector

def main():
    print("="*60)
    print("TESTING YOLO OBJECT DETECTION")
    print("="*60)
    
    # Initialize detector
    print("\nInitializing YOLO detector...")
    detector = ObjectDetector(model_name='yolov8n.pt', confidence=0.5)
    
    # Load image
    sequence_path = "data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 0
    
    print(f"\nLoading frame {frame_idx}...")
    left_img, right_img = load_kitti_stereo_pair(sequence_path, frame_idx)
    
    # Run detection
    print("\nRunning object detection...")
    start_time = time.time()
    detections = detector.detect(left_img)
    detect_time = time.time() - start_time
    
    print(f"\nDetection completed in {detect_time:.3f} seconds")
    print(f"FPS: {1.0/detect_time:.1f}")
    print(f"\nDetected {len(detections)} objects:")
    
    # Print detections
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']:15s} - confidence: {det['confidence']:.2f} - bbox: {det['bbox']}")
    
    # Draw detections
    print("\nDrawing bounding boxes...")
    annotated = detector.draw_detections(left_img, detections)
    
    # Display
    print("\nDisplaying results (close window to continue)...")
    
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(left_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f'Detected Objects ({len(detections)} found)')
    axes[1].axis('off')
    
    plt.suptitle(f'YOLO Object Detection - Frame {frame_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nTest completed successfully")
    
    return 0

if __name__ == "__main__":
    exit(main())
