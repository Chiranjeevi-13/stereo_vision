import cv2
import numpy as np
import os

def load_kitti_stereo_pair(sequence_path, frame_idx=0):
    """
    Load a stereo pair from KITTI dataset.
    
    Args:
        sequence_path: Path to sequence folder (e.g., 'data/kitti/2011_09_26/2011_09_26_drive_0001_sync')
        frame_idx: Frame number to load (0-107 for sequence 0001)
        
    Returns:
        left_img: Left color image (H x W x 3) BGR
        right_img: Right color image (H x W x 3) BGR
    """
    # Format frame number as 10-digit string with leading zeros
    frame_str = f"{frame_idx:010d}"
    
    # Construct paths to left and right images
    left_path = os.path.join(sequence_path, 'image_02', 'data', f'{frame_str}.png')
    right_path = os.path.join(sequence_path, 'image_03', 'data', f'{frame_str}.png')
    
    # Check files exist
    if not os.path.exists(left_path):
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not os.path.exists(right_path):
        raise FileNotFoundError(f"Right image not found: {right_path}")
    
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    # Check if loaded successfully
    if left_img is None:
        raise ValueError(f"Failed to load left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Failed to load right image: {right_path}")
    
    return left_img, right_img


def display_stereo_pair(left_img, right_img, title="KITTI Stereo Pair"):
    """
    Display stereo pair side-by-side using matplotlib.
    
    Args:
        left_img: Left image (BGR format from OpenCV)
        right_img: Right image (BGR format from OpenCV)
        title: Window title
    """
    import matplotlib.pyplot as plt
    
    # Convert BGR to RGB for matplotlib
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].imshow(left_rgb)
    axes[0].set_title('Left Camera (image_02)')
    axes[0].axis('off')
    
    axes[1].imshow(right_rgb)
    axes[1].set_title('Right Camera (image_03)')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def print_image_info(img, name="Image"):
    """
    Print image properties.
    
    Args:
        img: Image array
        name: Image name for display
    """
    print(f"\n{name} Properties:")
    print(f"  Shape: {img.shape} (Height x Width x Channels)")
    print(f"  Data type: {img.dtype}")
    print(f"  Value range: [{img.min()}, {img.max()}]")
    print(f"  Memory size: {img.nbytes / (1024*1024):.2f} MB")
