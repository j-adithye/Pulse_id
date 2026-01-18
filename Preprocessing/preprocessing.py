#!/usr/bin/env python3
"""
Step 1: Gaussian Blur
Step 2: Hand Segmentation

white pixel change cheyyan arn plan but moonji so
tommorow oru vignette add cheyya white ath chelpo ang povm
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

def segment_hand(img):
    """
    Segment hand from background using Otsu thresholding
    
    Args:
        img: Blurred grayscale image
    
    Returns:
        mask: Binary mask of hand region (255=hand, 0=background)
        segmented: Hand image with background removed
    """
    """
    Segment hand from background using Otsu thresholding
    ...
    """
    
    # NEW: Remove bright pixels (background) first
    # Pixels above this threshold = background (set to black)
    brightness_threshold = 113 # Adjust this (100-200)
    
    # Set bright pixels to 0 (black)
    img_filtered = img.copy()
    img_filtered[img_filtered > brightness_threshold] = 0
    
    print(f"✓ Removed pixels brighter than {brightness_threshold}")
    
    # Apply Otsu's thresholding (on filtered image)
    _, binary = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"✓ Otsu threshold applied")
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Close small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    print(f"✓ Morphological operations applied")
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠ Warning: No contours found, using full image")
        return np.ones_like(img) * 255, img
    
    # Get largest contour (should be the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Create clean mask from largest contour
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Apply mask to original image
    segmented = cv2.bitwise_and(img, img, mask=mask)
    
    print(f"✓ Hand segmented (contour area: {area:.0f} pixels)")
    
    return mask, segmented

def visualize_segmentation(original, blurred, mask, segmented):
    """Show segmentation process"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('2. Gaussian Blur (11×11)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('3. Hand Mask (Binary)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(segmented, cmap='gray')
    axes[1, 1].set_title('4. Segmented Hand')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Image paths
    image_path = "C://Users//adith//OneDrive//Documents//py//Pulse_id//image//vein.jpg"
    output_path = "C://Users//adith//OneDrive//Documents//py//Pulse_id//image//vein_processed.jpg"
    
    print("="*50)
    print("Step 1: Gaussian Blur")
    print("Step 2: Hand Segmentation")
    print("="*50)
    
    # Load image
    img_original = load_image(image_path)
    
    # Step 1: Apply Gaussian blur with 11×11 kernel
    img_blurred = apply_gaussian_blur(img_original, kernel_size=11)
    
    # Step 2: Segment hand from background
    print("\nSegmenting hand...")
    mask, img_segmented = segment_hand(img_blurred)
    
    # Visualize results
    visualize_segmentation(img_original, img_blurred, mask, img_segmented)
    
    # Save result
    cv2.imwrite(output_path, img_segmented)
    print(f"\n✓ Saved segmented image: {output_path}")