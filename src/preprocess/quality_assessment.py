"""
Quality assessment and filtering for fundus images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)


class FundusQualityAssessor:
    """Quality assessment for fundus images."""
    
    def __init__(self, quality_threshold: float = 0.8):
        """
        Initialize quality assessor.
        
        Args:
            quality_threshold: Minimum quality score to pass
        """
        self.quality_threshold = quality_threshold
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess the quality of a fundus image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing quality metrics and overall score
        """
        if image is None or image.size == 0:
            return {"score": 0.0, "passed": False, "reason": "Invalid image"}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        metrics = {}
        
        # 1. Sharpness (Laplacian variance)
        metrics["sharpness"] = self._calculate_sharpness(gray)
        
        # 2. Contrast
        metrics["contrast"] = self._calculate_contrast(gray)
        
        # 3. Brightness
        metrics["brightness"] = self._calculate_brightness(gray)
        
        # 4. Noise level
        metrics["noise"] = self._calculate_noise(gray)
        
        # 5. Focus quality (using FFT)
        metrics["focus"] = self._calculate_focus_quality(gray)
        
        # 6. Vessel visibility
        metrics["vessel_visibility"] = self._calculate_vessel_visibility(gray)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(metrics)
        metrics["overall_score"] = overall_score
        metrics["passed"] = overall_score >= self.quality_threshold
        
        return metrics
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate image contrast."""
        return np.std(gray)
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate image brightness."""
        return np.mean(gray)
    
    def _calculate_noise(self, gray: np.ndarray) -> float:
        """Calculate noise level using high-frequency components."""
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        return np.std(filtered)
    
    def _calculate_focus_quality(self, gray: np.ndarray) -> float:
        """Calculate focus quality using FFT."""
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Focus quality is related to high-frequency content
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # High-frequency region (outer area)
        high_freq = magnitude_spectrum[center_h//2:3*center_h//2, center_w//2:3*center_w//2]
        
        return np.mean(high_freq)
    
    def _calculate_vessel_visibility(self, gray: np.ndarray) -> float:
        """Calculate vessel visibility using edge detection."""
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels (vessels)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        
        return edge_pixels / total_pixels
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        # Normalize metrics to 0-1 range
        normalized = {}
        
        # Sharpness: higher is better, normalize to 0-1
        normalized["sharpness"] = min(metrics["sharpness"] / 1000, 1.0)
        
        # Contrast: higher is better, normalize to 0-1
        normalized["contrast"] = min(metrics["contrast"] / 100, 1.0)
        
        # Brightness: optimal range around 128, penalize extremes
        brightness = metrics["brightness"]
        if 64 <= brightness <= 192:
            normalized["brightness"] = 1.0
        else:
            normalized["brightness"] = max(0, 1 - abs(brightness - 128) / 128)
        
        # Noise: lower is better, invert and normalize
        normalized["noise"] = max(0, 1 - metrics["noise"] / 50)
        
        # Focus: higher is better, normalize to 0-1
        normalized["focus"] = min(metrics["focus"] / 10, 1.0)
        
        # Vessel visibility: higher is better, normalize to 0-1
        normalized["vessel_visibility"] = min(metrics["vessel_visibility"] * 100, 1.0)
        
        # Weighted average
        weights = {
            "sharpness": 0.25,
            "contrast": 0.20,
            "brightness": 0.15,
            "noise": 0.20,
            "focus": 0.15,
            "vessel_visibility": 0.05
        }
        
        overall_score = sum(normalized[key] * weights[key] for key in weights)
        return overall_score
    
    def filter_images(self, image_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Filter images based on quality assessment.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (passed_images, failed_images)
        """
        passed_images = []
        failed_images = []
        
        for img_path in image_paths:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    failed_images.append(img_path)
                    continue
                
                # Assess quality
                quality_metrics = self.assess_image_quality(image)
                
                if quality_metrics["passed"]:
                    passed_images.append(img_path)
                    logger.debug(f"Image {img_path.name} passed quality check: {quality_metrics['overall_score']:.3f}")
                else:
                    failed_images.append(img_path)
                    logger.debug(f"Image {img_path.name} failed quality check: {quality_metrics['overall_score']:.3f}")
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                failed_images.append(img_path)
        
        logger.info(f"Quality filtering complete: {len(passed_images)} passed, {len(failed_images)} failed")
        return passed_images, failed_images
