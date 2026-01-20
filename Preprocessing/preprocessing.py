#!/usr/bin/env python3
"""
Step 1: Gaussian Blur
Step 2: Rectangle Crop (Fast)
Step 3: GrabCut Segmentation (On cropped region - faster)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load image in grayscale"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"✓ Image loaded: {img.shape}")
    return img

def apply_gaussian_blur(img, kernel_size=11):
    """
    Apply Gaussian blur to reduce noise
    
    Args:
        img: Input grayscale image
        kernel_size: Blur kernel size (must be odd number)
    
    Returns:
        Blurred image
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    print(f"✓ Gaussian blur applied with kernel size: {kernel_size}×{kernel_size}")
    return blurred

def crop_hand_region(img):
    """
    Crop to hand region using fixed rectangle
    This removes most background and speeds up GrabCut
    
    Args:
        img: Blurred grayscale image
    
    Returns:
        cropped: Cropped hand region
        crop_coords: (x, y, width, height) for reference
    """
    import time
    start_time = time.time()
    
    height, width = img.shape
    
    # Define crop region (adjust these percentages based on your setup)
    x_start = int(width * 0.15)   # Start at 15% from left
    x_end = int(width * 0.85)     # End at 85% (keep 70% width)
    y_start = int(height * 0.15)  # Start at 15% from top  
    y_end = int(height * 0.85)    # End at 85% (keep 70% height)
    
    # Crop the image
    cropped = img[y_start:y_end, x_start:x_end]
    
    crop_coords = (x_start, y_start, x_end - x_start, y_end - y_start)
    
    elapsed = time.time() - start_time
    print(f"✓ Rectangle crop complete in {elapsed:.4f} seconds")
    print(f"  Original: {width}×{height} → Cropped: {cropped.shape[1]}×{cropped.shape[0]}")
    
    return cropped, crop_coords

def segment_hand_grabcut(img):
    """
    Segment hand using GrabCut algorithm
    Runs on already-cropped image so it's faster
    
    Args:
        img: Cropped grayscale image
    
    Returns:
        mask: Binary mask of hand region
        segmented: Hand image with background removed
    """
    import time
    start_time = time.time()
    
    # GrabCut needs RGB image
    img_3channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Create a mask for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Define rectangle (use most of the cropped region)
    height, width = img.shape
    margin_x = int(width * 0.05)   # 5% margin
    margin_y = int(height * 0.05)  # 5% margin
    
    rect = (margin_x, margin_y, 
            width - 2*margin_x, 
            height - 2*margin_y)
    
    # Background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    print("Running GrabCut on cropped region...")
    
    # Run GrabCut (fewer iterations since we already cropped)
    cv2.grabCut(img_3channel, mask, rect, bgd_model, fgd_model, 
                3, cv2.GC_INIT_WITH_RECT)  # 3 iterations (was 5)
    
    # Create binary mask
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    binary_mask = binary_mask * 255
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    segmented = cv2.bitwise_and(img, img, mask=binary_mask)
    
    elapsed = time.time() - start_time
    print(f"✓ GrabCut segmentation complete in {elapsed:.2f} seconds")
    
    return binary_mask, segmented

def normalize_brightness(img):
    """
    Normalize brightness for saving
    Ensures saved image matches visualization brightness
    
    Args:
        img: Input image
    
    Returns:
        Normalized image (0-255 range, uint8)
    """
    # If image is float, convert to uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_min = np.min(img)
        img_max = np.max(img)
        
        if img_max > img_min:
            # Normalize to 0-255
            normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = img.astype(np.uint8)
    else:
        normalized = img
    
    # Apply histogram equalization for better visibility
    # This matches what matplotlib does for display
    normalized = cv2.equalizeHist(normalized)
    
    print(f"✓ Brightness normalized for saving")
    return normalized

def visualize_pipeline(original, blurred, cropped, mask, segmented):
    """Show full preprocessing pipeline"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('2. Gaussian Blur')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cropped, cmap='gray')
    axes[0, 2].set_title('3. Rectangle Crop')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('4. GrabCut Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(segmented, cmap='gray')
    axes[1, 1].set_title('5. Segmented Hand')
    axes[1, 1].axis('off')
    
    # Show final normalized version (what will be saved)
    segmented_normalized = normalize_brightness(segmented)
    axes[1, 2].imshow(segmented_normalized, cmap='gray')
    axes[1, 2].set_title('6. Normalized (Saved)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Image paths
    image_path = "C://Users//adith//OneDrive//Documents//py//Pulse_id//image//vein.jpg"
    output_path = "C://Users//adith//OneDrive//Documents//py//Pulse_id//image//vein_processed.jpg"
    
    print("="*60)
    print("Palm Vein Preprocessing Pipeline")
    print("Step 1: Gaussian Blur")
    print("Step 2: Rectangle Crop (removes most background)")
    print("Step 3: GrabCut Segmentation (on cropped region - faster)")
    print("="*60)
    
    # Load image
    img_original = load_image(image_path)
    
    # Step 1: Gaussian blur
    img_blurred = apply_gaussian_blur(img_original, kernel_size=11)
    
    # Step 2: Rectangle crop (fast)
    print("\nCropping to hand region...")
    img_cropped, crop_coords = crop_hand_region(img_blurred)
    
    # Step 3: GrabCut segmentation (on cropped image - faster!)
    print("\nSegmenting hand with GrabCut...")
    mask, img_segmented = segment_hand_grabcut(img_cropped)
    
    # Normalize brightness before saving
    img_final = normalize_brightness(img_segmented)
    
    # Visualize all steps
    visualize_pipeline(img_original, img_blurred, img_cropped, mask, img_final)
    
    # Save normalized result
    cv2.imwrite(output_path, img_final)
    print(f"\n✓ Saved final image: {output_path}")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("Expected speed: ~20-40 seconds per image")
    print("(Much faster than 80s due to cropping first)")
    print("="*60)