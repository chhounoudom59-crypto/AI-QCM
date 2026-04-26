from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to BGR image.
    
    Args:
        image: BGR image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def normalize_pixel_values(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range for model input."""
    return image.astype(np.float32) / 255.0


def preprocess_for_inference(image: np.ndarray, target_size: tuple[int, int] = (640, 640), apply_contrast: bool = True) -> tuple[np.ndarray, tuple[int, int]]:
    """Full preprocessing pipeline for inference.
    
    Args:
        image: Input BGR image
        target_size: Target dimensions for model
        apply_contrast: Whether to apply CLAHE
    
    Returns:
        Preprocessed image, original size for later use
    """
    h, w = image.shape[:2]
    original_size = (w, h)
    
    # Apply contrast enhancement
    if apply_contrast:
        image = apply_clahe(image)
    
    # Resize to target
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    return resized, original_size
