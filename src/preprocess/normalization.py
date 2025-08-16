"""
Image normalization and preprocessing for fundus images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)


class FundusNormalizer:
    """Normalize and preprocess fundus images."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalization: str = "imagenet",
        augmentation: bool = False
    ):
        """
        Initialize normalizer.
        
        Args:
            target_size: Target image dimensions (height, width)
            normalization: Normalization method ("imagenet", "standard", "minmax")
            augmentation: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.normalization = normalization
        self.augmentation = augmentation
        
        # ImageNet normalization constants
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
    
    def normalize_image(
        self,
        image: np.ndarray,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Normalize a fundus image.
        
        Args:
            image: Input image as numpy array (BGR format)
            output_path: Optional path to save normalized image
            
        Returns:
            Normalized image as numpy array
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing pipeline
        processed = self._preprocess_pipeline(image_rgb)
        
        # Apply normalization
        normalized = self._apply_normalization(processed)
        
        # Save if output path provided
        if output_path:
            self._save_normalized_image(normalized, output_path)
        
        return normalized
    
    def _preprocess_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to image."""
        # 1. Resize to target dimensions
        resized = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Convert to LAB color space
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 3. Apply data augmentation if enabled
        if self.augmentation:
            enhanced = self._apply_augmentation(enhanced)
        
        return enhanced
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques."""
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            enhanced = Image.fromarray(image)
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(factor)
            image = np.array(enhanced)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            enhanced = Image.fromarray(image)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(factor)
            image = np.array(enhanced)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (small angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return image
    
    def _apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization to image."""
        # Convert to float32 and scale to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        if self.normalization == "imagenet":
            # ImageNet normalization
            normalized = (image_float - self.imagenet_mean) / self.imagenet_std
        elif self.normalization == "standard":
            # Standard normalization (zero mean, unit variance)
            mean = np.mean(image_float, axis=(0, 1))
            std = np.std(image_float, axis=(0, 1))
            normalized = (image_float - mean) / (std + 1e-8)
        elif self.normalization == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            normalized = (image_float - min_val) / (max_val - min_val + 1e-8)
        else:
            # No normalization
            normalized = image_float
        
        return normalized
    
    def _save_normalized_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save normalized image to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert back to uint8 for saving
        if self.normalization == "imagenet":
            # Denormalize ImageNet
            denormalized = (image * self.imagenet_std + self.imagenet_mean) * 255
        elif self.normalization == "standard":
            # Denormalize standard
            denormalized = image * 255
        elif self.normalization == "minmax":
            # Denormalize minmax
            denormalized = image * 255
        else:
            denormalized = image * 255
        
        # Clip to valid range
        denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        denormalized_bgr = cv2.cvtColor(denormalized, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(str(output_path), denormalized_bgr)
        logger.debug(f"Saved normalized image to {output_path}")
    
    def batch_normalize(
        self,
        input_paths: list,
        output_dir: Path,
        quality_passed: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Normalize multiple images in batch.
        
        Args:
            input_paths: List of input image paths
            output_dir: Output directory for normalized images
            quality_passed: Optional list of quality-passed image paths
            
        Returns:
            Dictionary with processing statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total": len(input_paths),
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        for img_path in input_paths:
            try:
                # Skip if not in quality-passed list
                if quality_passed is not None and img_path not in quality_passed:
                    continue
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    stats["failed"] += 1
                    stats["errors"].append(f"Failed to load {img_path}")
                    continue
                
                # Generate output path
                output_path = output_dir / f"norm_{img_path.stem}.png"
                
                # Normalize image
                self.normalize_image(image, output_path)
                stats["processed"] += 1
                
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"Error processing {img_path}: {e}")
                logger.error(f"Error processing {img_path}: {e}")
        
        logger.info(f"Batch normalization complete: {stats['processed']} processed, {stats['failed']} failed")
        return stats
