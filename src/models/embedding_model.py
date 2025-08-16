"""
CNN embedding model for fundus image feature extraction.
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class FundusEmbeddingModel:
    """CNN model for extracting embeddings from fundus images."""
    
    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        embedding_dim: int = 1280,
        pretrained: bool = True,
        device: str = "auto"
    ):
        """
        Initialize embedding model.
        
        Args:
            architecture: Model architecture from timm
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained weights
            device: Device to run model on ("auto", "cpu", "cuda")
        """
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.device = self._setup_device(device)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized {architecture} model on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for model."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def _create_model(self) -> nn.Module:
        """Create the embedding model."""
        try:
            # Create model from timm
            model = timm.create_model(
                self.architecture,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head
                global_pool="avg"  # Global average pooling
            )
            
            # Verify output dimension
            if hasattr(model, 'num_features'):
                actual_dim = model.num_features
            else:
                # Test with dummy input to get output dimension
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    output = model(dummy_input)
                    actual_dim = output.shape[1]
            
            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Model output dimension ({actual_dim}) doesn't match "
                    f"expected ({self.embedding_dim}). Using actual dimension."
                )
                self.embedding_dim = actual_dim
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {self.architecture}: {e}")
            raise
    
    def extract_embedding(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract embedding from a single image.
        
        Args:
            image: Input image as numpy array (normalized, RGB format)
            normalize: Whether to normalize the output embedding
            
        Returns:
            Image embedding as numpy array
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)
            embedding = embedding.cpu().numpy()
        
        # Remove batch dimension if single image
        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        
        # Normalize if requested
        if normalize:
            embedding = self._normalize_embedding(embedding)
        
        return embedding
    
    def extract_embeddings_batch(
        self,
        images: np.ndarray,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings from a batch of images.
        
        Args:
            images: Batch of images as numpy array (N, C, H, W)
            batch_size: Batch size for processing
            normalize: Whether to normalize output embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Batch of embeddings as numpy array (N, embedding_dim)
        """
        if images is None or images.size == 0:
            raise ValueError("Invalid input images")
        
        num_images = images.shape[0]
        embeddings = []
        
        # Process in batches
        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            batch_images = images[i:batch_end]
            
            # Convert to tensor
            batch_tensor = torch.from_numpy(batch_images).float().to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                batch_embeddings = self._normalize_embeddings(batch_embeddings)
            
            embeddings.append(batch_embeddings)
            
            if show_progress:
                logger.info(f"Processed {batch_end}/{num_images} images")
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info(f"Extracted embeddings for {num_images} images")
        return all_embeddings
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a single embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize a batch of embedding vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return embeddings / norms
    
    def save_model(self, output_path: Path) -> None:
        """Save the model to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'architecture': self.architecture,
            'embedding_dim': self.embedding_dim,
            'model_state_dict': self.model.state_dict(),
            'pretrained': self.pretrained
        }, output_path)
        
        logger.info(f"Model saved to {output_path}")
    
    @classmethod
    def load_model(cls, model_path: Path, device: str = "auto") -> 'FundusEmbeddingModel':
        """Load a saved model from disk."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = cls(
            architecture=checkpoint['architecture'],
            embedding_dim=checkpoint['embedding_dim'],
            pretrained=False,  # Already loaded weights
            device=device
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'architecture': self.architecture,
            'embedding_dim': self.embedding_dim,
            'pretrained': self.pretrained,
            'device': str(self.device),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
