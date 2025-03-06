import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import tifffile  # Alternative for multi-band TIFF files


def preprocess_image(filepath):
    """
    Preprocess the multi-band TIF image using tifffile instead of rasterio.
    """
    # Read TIF file with tifffile
    image = tifffile.imread(filepath)
    
    # Check if image is in expected format
    if len(image.shape) == 3 and image.shape[2] >= 5:
        # Extract RGB channels (2:Blue, 3:Green, 4:Red) - using 0-based indexing
        rgb_image = image[:, :, [2, 3, 4]]
    else:
        # If image doesn't have expected channels, try to handle gracefully
        print(f"Warning: Image shape {image.shape} doesn't match expected format. Using available channels.")
        if len(image.shape) == 3:
            # Use what we have, limited to 3 channels for RGB
            num_channels = min(3, image.shape[2])
            rgb_image = image[:, :, :num_channels]
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            rgb_image = np.stack([image, image, image], axis=2)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Ensure we have float values
    if rgb_image.dtype == np.uint16:
        # 16-bit images need to be scaled to 0-1
        rgb_image = rgb_image.astype(np.float32) / 65535.0
    elif rgb_image.dtype == np.uint8:
        # 8-bit images need to be scaled to 0-1
        rgb_image = rgb_image.astype(np.float32) / 255.0
    else:
        # Already float or other type
        rgb_image = rgb_image.astype(np.float32)
        if np.max(rgb_image) > 1.0:
            rgb_image = rgb_image / np.max(rgb_image)
    
    # Add batch dimension
    rgb_image = np.expand_dims(rgb_image, axis=0)
    
    return rgb_image


def postprocess_prediction(prediction, filename):
    """
    Process the model's prediction and save visualization.
    """
    # Convert prediction to binary mask
    binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_mask, cmap='binary')
    plt.axis('off')
    
    # Save visualization
    result_path = os.path.join('static/uploads', f'pred_{os.path.basename(filename)}.png')
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return result_path


def overlay_prediction(original_path, prediction, filename):
    """
    Create an overlay of the original image and the prediction.
    """
    # Read the original image using tifffile
    try:
        image = tifffile.imread(original_path)
        
        # Extract RGB channels for visualization
        if len(image.shape) == 3 and image.shape[2] >= 5:
            rgb_image = image[:, :, [2, 3, 4]]
        elif len(image.shape) == 3:
            # Use what we have, limited to 3 channels for RGB
            num_channels = min(3, image.shape[2])
            rgb_image = image[:, :, :num_channels]
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            rgb_image = np.stack([image, image, image], axis=2)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
            
        # Normalize for visualization
        if rgb_image.dtype == np.uint16:
            rgb_image = rgb_image.astype(np.float32) / 65535.0
        elif rgb_image.dtype == np.uint8:
            rgb_image = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_image = rgb_image.astype(np.float32)
            if np.max(rgb_image) > 1.0:
                rgb_image = rgb_image / np.max(rgb_image)
    except Exception as e:
        print(f"Error reading original image for overlay: {str(e)}")
        # Create a blank image if we can't read the original
        rgb_image = np.ones((prediction.shape[1], prediction.shape[2], 3), dtype=np.float32)
    
    # Get binary mask
    binary_mask = (prediction[0, :, :, 0] > 0.5)
    
    # Create overlay
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Prediction mask
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='binary')
    plt.title("Water Mask")
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_image)
    plt.imshow(binary_mask, alpha=0.4, cmap='Blues')
    plt.title("Overlay")
    plt.axis('off')
    
    # Save visualization
    overlay_path = os.path.join('static/uploads', f'overlay_{os.path.basename(filename)}.png')
    plt.savefig(overlay_path, bbox_inches='tight')
    plt.close()
    
    return overlay_path