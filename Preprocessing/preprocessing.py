#!/usr/bin/env python3
"""
Step 1: Gaussian Blur
Step 2: Rectangle Crop (Fast)
Step 3: Otsu Thresholding Segmentation
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

def segment_hand_otsu(img):
    """
    Segment hand using Otsu thresholding with contour filtering
    Keeps only the largest region (hand), removes noise
    
    Args:
        img: Cropped grayscale image
    
    Returns:
        mask: Binary mask of hand region
        segmented: Hand image with background removed
    """
    import time
    start_time = time.time()
    
    print("Running Otsu thresholding...")
    
    # Apply Otsu's thresholding (inverted for NIR)
    _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours to remove noise
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Keep only the largest contour (hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create clean mask with only the largest contour
        clean_mask = np.zeros_like(binary_mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        binary_mask = clean_mask
        print(f"✓ Kept largest contour, removed noise")
    
    # Apply mask to original image
    segmented = cv2.bitwise_and(img, img, mask=binary_mask)
    
    elapsed = time.time() - start_time
    print(f"✓ Otsu segmentation complete in {elapsed:.4f} seconds")
    
    return binary_mask, segmented

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
    axes[1, 0].set_title('4. Otsu Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(segmented, cmap='gray')
    axes[1, 1].set_title('5. Segmented Hand')
    axes[1, 1].axis('off')
    
    # Hide the 6th subplot
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
    print("Step 3: Otsu Thresholding (fast segmentation)")
    print("="*60)
    
    # Load image
    img_original = load_image(image_path)
    
    # Step 1: Gaussian blur
    img_blurred = apply_gaussian_blur(img_original, kernel_size=11)
    
    # Step 2: Rectangle crop (fast)
    print("\nCropping to hand region...")
    img_cropped, crop_coords = crop_hand_region(img_blurred)
    
    # Step 3: Otsu thresholding segmentation (fast!)
    print("\nSegmenting hand with Otsu...")
    mask, img_segmented = segment_hand_otsu(img_cropped)
    
    # Visualize all steps
    visualize_pipeline(img_original, img_blurred, img_cropped, mask, img_segmented)
    
    # Save result
    cv2.imwrite(output_path, img_segmented)
    print(f"\n✓ Saved segmented image: {output_path}")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("Expected speed: <1 second per image")
    print("="*60)