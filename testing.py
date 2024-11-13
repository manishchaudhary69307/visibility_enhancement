import numpy as np
import cv2
import matplotlib.pyplot as plt


# Fuzzy Membership Function for Intensification
def fuzzy_intensification(image, low_thresh, mid_thresh, high_thresh):
    # Normalize the image to the range [0, 1] for easier manipulation
    image_normalized = image / 255.0
    print(image_normalized)

    # Define fuzzy membership functions for three regions (low, medium, high)
    # Membership for low intensity region (inverse of image intensity)
    low_membership = np.clip((low_thresh - image_normalized) / low_thresh, 0, 1)
    print(low_membership,"low members")

    # Membership for middle intensity region (bell-shaped curve around mid_thresh)
    mid_membership = np.clip(1 - np.abs(image_normalized - mid_thresh / 255.0) / ((high_thresh - low_thresh) / 255.0),
                             0, 1)

    # Membership for high intensity region
    high_membership = np.clip((image_normalized - high_thresh / 255.0) / (1 - high_thresh / 255.0), 0, 1)

    return low_membership, mid_membership, high_membership


# Fuzzy Intensification operator to enhance image intensity based on fuzzy memberships
def fuzzy_enhance(image, low_membership, mid_membership, high_membership, enhancement_factor=1.5):
    # Normalize image to range [0, 1]
    image_normalized = image / 255.0

    # Calculate intensification for each region based on the fuzzy memberships
    enhanced_image = np.zeros_like(image, dtype=np.float32)

    # Enhance low intensity regions
    enhanced_image += low_membership * image_normalized * enhancement_factor  # Enhance dark regions
    # Enhance mid intensity regions (moderate enhancement)
    enhanced_image += mid_membership * image_normalized * (enhancement_factor * 1.3)  # Moderate enhancement
    # Enhance high intensity regions (slightly less enhancement to preserve details)
    enhanced_image += high_membership * image_normalized * enhancement_factor * 0.8  # Less enhancement for bright regions

    # Normalize back to 0-255 and ensure no overflow
    enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)

    return enhanced_image


# Tri-threshold Fuzzy Intensification enhancement function
def tri_threshold_fuzzy_intensification(image_path, low_thresh, mid_thresh, high_thresh, enhancement_factor=1.5):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found. Please check the file path.")

    # Fuzzy membership functions
    low_membership, mid_membership, high_membership = fuzzy_intensification(img, low_thresh, mid_thresh, high_thresh)

    # Apply fuzzy intensification
    enhanced_img = fuzzy_enhance(img, low_membership, mid_membership, high_membership, enhancement_factor)

    return enhanced_img


# Function to visualize results
def show_results(original, enhanced):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.show()


# Main execution
if __name__ == "__main__":
    image_path = 'images/dusty2.jpg'  # Replace with the actual image path

    # Define thresholds for tri-threshold segmentation
    low_thresh = 10  # Low intensity threshold
    mid_thresh = 20  # Mid intensity threshold
    high_thresh = 80  # High intensity threshold
    enhancement_factor = 0.5  # Define enhancement factor for intensification (less aggressive)

    # Perform the tri-threshold fuzzy intensification
    enhanced_image = tri_threshold_fuzzy_intensification(image_path, low_thresh, mid_thresh, high_thresh,
                                                         enhancement_factor)

    # Load original image for comparison
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Show the original and enhanced images
    show_results(original_image, enhanced_image)
