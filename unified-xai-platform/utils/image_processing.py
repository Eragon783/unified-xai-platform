"""
Image Processing Utilities
Handles X-ray image loading and preprocessing for lung cancer detection.
"""

import numpy as np
from PIL import Image
import cv2


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(file_path: str) -> Image.Image:
    """
    Load an image file.

    Args:
        file_path: Path to the image file

    Returns:
        PIL Image object
    """
    return Image.open(file_path).convert('RGB')


def preprocess_xray(image_path: str, target_size: tuple = (224, 224), normalize: bool = True) -> np.ndarray:
    """
    Preprocess chest X-ray image for model input.

    Args:
        image_path: Path to the X-ray image
        target_size: Expected input size for the model
        normalize: Whether to apply ImageNet normalization

    Returns:
        Preprocessed numpy array ready for model prediction
    """
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img) / 255.0

    if normalize:
        # Apply ImageNet normalization
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)


def preprocess_image_simple(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Simple preprocessing without ImageNet normalization.
    Used for XAI visualizations.

    Args:
        image_path: Path to the image
        target_size: Target size

    Returns:
        Preprocessed numpy array
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a heatmap on the original image.

    Args:
        original_image: Original image array (H, W, 3)
        heatmap: Heatmap array (H, W)
        alpha: Transparency factor for the heatmap

    Returns:
        Combined image with heatmap overlay
    """
    # Normalize heatmap to [0, 255]
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure original image is in the right format
    if original_image.max() <= 1.0:
        original_image = np.uint8(255 * original_image)

    # Resize heatmap to match original image
    heatmap_colored = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))

    # Blend images
    superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed


def get_image_info(file_path: str) -> dict:
    """
    Get information about an image file.

    Args:
        file_path: Path to the image file

    Returns:
        Dictionary with image information
    """
    img = Image.open(file_path)
    return {
        'format': img.format,
        'mode': img.mode,
        'size': img.size,
        'width': img.width,
        'height': img.height
    }


def load_image_from_upload(uploaded_file, temp_dir: str, target_size: tuple = (224, 224)) -> tuple:
    """
    Load and preprocess image from Streamlit uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object
        temp_dir: Directory to save temporary files
        target_size: Target size for the model

    Returns:
        Tuple of (image_data, image_path)
    """
    import os

    # Save uploaded file temporarily
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(target_size)

    return image_resized, image_path


def image_to_array(image_data, normalize: bool = True) -> np.ndarray:
    """
    Convert image to numpy array for model input.

    Args:
        image_data: PIL Image
        normalize: Whether to normalize to [0, 1]

    Returns:
        Numpy array ready for model prediction
    """
    img_array = np.array(image_data)
    if normalize:
        img_array = img_array / 255.0
    return img_array
