#!/usr/bin/env python3
"""
Test KITTI stereo image loader
"""
from utils.loader import load_kitti_stereo_pair, display_stereo_pair, print_image_info

def main():
    print("="*60)
    print("TESTING KITTI STEREO LOADER")
    print("="*60)
    
    # Define path to first sequence
    sequence_path = "data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    frame_idx = 0
    
    print(f"\nLoading frame {frame_idx} from sequence 0001...")
    
    try:
        # Load stereo pair
        left_img, right_img = load_kitti_stereo_pair(sequence_path, frame_idx)
        
        print("\nSuccessfully loaded stereo pair")
        
        # Print image information
        print_image_info(left_img, "Left Image")
        print_image_info(right_img, "Right Image")
        
        # Display images
        print("\nDisplaying images (close window to continue)...")
        display_stereo_pair(left_img, right_img, f"KITTI Sequence 0001 - Frame {frame_idx}")
        
        print("\nTest completed successfully")
        print("\nNext: Try different frames (0-107) or sequences (0005, 0009)")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
