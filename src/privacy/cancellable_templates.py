"""
Privacy-preserving cancellable biometric templates.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import logging
import pickle

logger = logging.getLogger(__name__)


class CancellableTemplateGenerator:
    """Generate cancellable biometric templates for privacy protection."""
    
    def __init__(
        self,
        template_dim: int = 512,
        random_projection_seed: int = 42,
        sign_code_bits: int = 256,
        cancellable: bool = True
    ):
        """
        Initialize cancellable template generator.
        
        Args:
            template_dim: Dimension of output templates
            random_projection_seed: Random seed for projection matrix
            sign_code_bits: Number of bits for sign codes
            cancellable: Whether to generate cancellable templates
        """
        self.template_dim = template_dim
        self.random_projection_seed = random_projection_seed
        self.sign_code_bits = sign_code_bits
        self.cancellable = cancellable
        
        # Set random seed for reproducibility
        np.random.seed(random_projection_seed)
        
        # Initialize projection matrix
        self.projection_matrix = None
        self._initialize_projection_matrix()
        
        logger.info(f"Initialized cancellable template generator with dim={template_dim}")
    
    def _initialize_projection_matrix(self) -> None:
        """Initialize random projection matrix."""
        if self.cancellable:
            # Generate random projection matrix
            self.projection_matrix = np.random.randn(self.template_dim, 1280)  # Assuming EfficientNet-B0 output
            # Normalize rows
            self.projection_matrix = self.projection_matrix / np.linalg.norm(self.projection_matrix, axis=1, keepdims=True)
        else:
            # Identity projection (no privacy protection)
            self.projection_matrix = np.eye(1280, self.template_dim).T
    
    def generate_template(
        self,
        embedding: np.ndarray,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate cancellable template from embedding.
        
        Args:
            embedding: Input embedding vector
            user_id: Optional user identifier for template naming
            
        Returns:
            Dictionary containing template and metadata
        """
        if embedding is None or embedding.size == 0:
            raise ValueError("Invalid input embedding")
        
        # Apply random projection
        projected = self._apply_projection(embedding)
        
        # Generate sign codes
        sign_codes = self._generate_sign_codes(projected)
        
        # Create template
        template = {
            'template_vector': projected,
            'sign_codes': sign_codes,
            'template_dim': self.template_dim,
            'sign_code_bits': self.sign_code_bits,
            'cancellable': self.cancellable,
            'user_id': user_id,
            'metadata': {
                'projection_seed': self.random_projection_seed,
                'generation_timestamp': np.datetime64('now')
            }
        }
        
        return template
    
    def generate_templates_batch(
        self,
        embeddings: np.ndarray,
        user_ids: Optional[list] = None
    ) -> list:
        """
        Generate cancellable templates for multiple embeddings.
        
        Args:
            embeddings: Batch of embeddings (N, embedding_dim)
            user_ids: Optional list of user identifiers
            
        Returns:
            List of template dictionaries
        """
        if embeddings is None or embeddings.size == 0:
            raise ValueError("Invalid input embeddings")
        
        num_embeddings = embeddings.shape[0]
        templates = []
        
        for i in range(num_embeddings):
            user_id = user_ids[i] if user_ids else f"user_{i:06d}"
            template = self.generate_template(embeddings[i], user_id)
            templates.append(template)
        
        logger.info(f"Generated {num_embeddings} cancellable templates")
        return templates
    
    def _apply_projection(self, embedding: np.ndarray) -> np.ndarray:
        """Apply random projection to embedding."""
        # Ensure embedding is 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # Apply projection
        projected = np.dot(self.projection_matrix, embedding)
        
        # Normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        
        return projected
    
    def _generate_sign_codes(self, projected: np.ndarray) -> np.ndarray:
        """Generate sign codes from projected vector."""
        # Convert to binary using sign function
        signs = np.sign(projected)
        
        # Convert to binary (1 for positive, 0 for negative)
        binary_codes = (signs > 0).astype(np.uint8)
        
        # Take first sign_code_bits
        if len(binary_codes) > self.sign_code_bits:
            binary_codes = binary_codes[:self.sign_code_bits]
        elif len(binary_codes) < self.sign_code_bits:
            # Pad with zeros if needed
            padding = np.zeros(self.sign_code_bits - len(binary_codes), dtype=np.uint8)
            binary_codes = np.concatenate([binary_codes, padding])
        
        return binary_codes
    
    def regenerate_template(
        self,
        embedding: np.ndarray,
        new_seed: int,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Regenerate template with new random seed (cancellation).
        
        Args:
            embedding: Input embedding vector
            new_seed: New random seed for projection
            user_id: Optional user identifier
            
        Returns:
            New template dictionary
        """
        # Store original seed
        original_seed = self.random_projection_seed
        
        # Update seed and regenerate projection matrix
        self.random_projection_seed = new_seed
        np.random.seed(new_seed)
        self._initialize_projection_matrix()
        
        # Generate new template
        new_template = self.generate_template(embedding, user_id)
        
        # Restore original seed
        self.random_projection_seed = original_seed
        np.random.seed(original_seed)
        self._initialize_projection_matrix()
        
        logger.info(f"Regenerated template with new seed {new_seed}")
        return new_template
    
    def save_templates(
        self,
        templates: list,
        output_path: Path,
        format: str = "pickle"
    ) -> None:
        """
        Save templates to disk.
        
        Args:
            templates: List of template dictionaries
            output_path: Output file path
            format: Output format ("pickle" or "numpy")
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(templates, f)
        elif format == "numpy":
            # Save as separate arrays
            template_vectors = np.array([t['template_vector'] for t in templates])
            sign_codes = np.array([t['sign_codes'] for t in templates])
            
            np.savez(
                output_path,
                template_vectors=template_vectors,
                sign_codes=sign_codes,
                metadata=[t['metadata'] for t in templates]
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(templates)} templates to {output_path}")
    
    @classmethod
    def load_templates(
        cls,
        file_path: Path,
        format: str = "auto"
    ) -> list:
        """
        Load templates from disk.
        
        Args:
            file_path: Input file path
            format: Input format ("auto", "pickle", or "numpy")
            
        Returns:
            List of template dictionaries
        """
        if format == "auto":
            if file_path.suffix == ".pkl":
                format = "pickle"
            elif file_path.suffix == ".npz":
                format = "numpy"
            else:
                format = "pickle"  # Default
        
        if format == "pickle":
            with open(file_path, 'rb') as f:
                templates = pickle.load(f)
        elif format == "numpy":
            data = np.load(file_path, allow_pickle=True)
            templates = []
            
            for i in range(len(data['template_vectors'])):
                template = {
                    'template_vector': data['template_vectors'][i],
                    'sign_codes': data['sign_codes'][i],
                    'metadata': data['metadata'][i]
                }
                templates.append(template)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded {len(templates)} templates from {file_path}")
        return templates
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about the template generator."""
        return {
            'template_dim': self.template_dim,
            'random_projection_seed': self.random_projection_seed,
            'sign_code_bits': self.sign_code_bits,
            'cancellable': self.cancellable,
            'projection_matrix_shape': self.projection_matrix.shape if self.projection_matrix is not None else None
        }
