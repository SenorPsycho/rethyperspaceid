"""
FAISS index for efficient similarity search of biometric templates.
"""

import faiss
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
import pickle

logger = logging.getLogger(__name__)


class FundusFAISSIndex:
    """FAISS index for fundus biometric template search and clustering."""
    
    def __init__(
        self,
        index_type: str = "faiss",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
        dimension: Optional[int] = None
    ):
        """
        Initialize FAISS index.
        
        Args:
            index_type: Type of FAISS index
            metric: Distance metric ("cosine", "l2", "ip")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to probe during search
            use_gpu: Whether to use GPU acceleration
            dimension: Dimension of template vectors
        """
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.dimension = dimension
        
        # Initialize index
        self.index = None
        self.template_metadata = []
        self.is_trained = False
        
        # GPU resources
        self.gpu_resources = None
        if self.use_gpu:
            self._setup_gpu()
        
        logger.info(f"Initialized FAISS index: {index_type}, metric: {metric}")
    
    def _setup_gpu(self) -> None:
        """Setup GPU resources for FAISS."""
        try:
            ngpus = faiss.get_num_gpus()
            if ngpus > 0:
                self.gpu_resources = faiss.StandardGpuResources()
                logger.info(f"GPU acceleration enabled with {ngpus} GPUs")
            else:
                logger.warning("No GPUs available, falling back to CPU")
                self.use_gpu = False
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}, falling back to CPU")
            self.use_gpu = False
    
    def create_index(self, dimension: int) -> None:
        """
        Create FAISS index with specified dimension.
        
        Args:
            dimension: Dimension of template vectors
        """
        self.dimension = dimension
        
        if self.index_type == "faiss":
            if self.metric == "cosine":
                # Cosine similarity using L2 index with normalized vectors
                self.index = faiss.IndexFlatL2(dimension)
            elif self.metric == "l2":
                self.index = faiss.IndexFlatL2(dimension)
            elif self.metric == "ip":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        
        elif self.index_type == "ivf":
            # IVF index for larger datasets
            if self.metric == "cosine":
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            elif self.metric == "l2":
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            elif self.metric == "ip":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Move to GPU if enabled
        if self.use_gpu and self.gpu_resources is not None:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
        
        logger.info(f"Created {self.index_type} index with dimension {dimension}")
    
    def add_templates(
        self,
        templates: list,
        batch_size: int = 1000
    ) -> None:
        """
        Add templates to the index.
        
        Args:
            templates: List of template dictionaries
            batch_size: Batch size for adding templates
        """
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        if not templates:
            logger.warning("No templates provided")
            return
        
        # Extract template vectors
        template_vectors = []
        for template in templates:
            if 'template_vector' in template:
                template_vectors.append(template['template_vector'])
                self.template_metadata.append(template)
        
        if not template_vectors:
            logger.warning("No valid template vectors found")
            return
        
        # Convert to numpy array
        vectors = np.array(template_vectors, dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        
        # Add vectors to index
        if self.index_type == "ivf" and not self.is_trained:
            # Train IVF index
            self.index.train(vectors)
            self.is_trained = True
            logger.info("Trained IVF index")
        
        # Add vectors
        self.index.add(vectors)
        
        logger.info(f"Added {len(template_vectors)} templates to index")
    
    def search(
        self,
        query_template: Union[np.ndarray, dict],
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search for similar templates.
        
        Args:
            query_template: Query template vector or dictionary
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Search results dictionary
        """
        if self.index is None:
            raise RuntimeError("Index not created or empty.")
        
        # Extract query vector
        if isinstance(query_template, dict):
            query_vector = query_template['template_vector']
        else:
            query_vector = query_template
        
        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Set nprobe for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # Search
        distances, indices = self.index.search(query_vector, min(top_k, len(self.template_metadata)))
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            
            # Convert distance to similarity
            if self.metric == "cosine":
                similarity = 1.0 - distance / 2.0  # Convert L2 distance to cosine similarity
            elif self.metric == "l2":
                similarity = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity
            elif self.metric == "ip":
                similarity = distance  # Inner product is already similarity
            
            # Filter by threshold
            if similarity >= similarity_threshold:
                result = {
                    'rank': i + 1,
                    'index': int(idx),
                    'distance': float(distance),
                    'similarity': float(similarity),
                    'template': self.template_metadata[idx]
                }
                results.append(result)
        
        return {
            'query_template': query_template,
            'results': results,
            'total_results': len(results),
            'search_params': {
                'top_k': top_k,
                'similarity_threshold': similarity_threshold,
                'metric': self.metric
            }
        }
    
    def batch_search(
        self,
        query_templates: list,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for multiple query templates.
        
        Args:
            query_templates: List of query template dictionaries
            top_k: Number of top results per query
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search result dictionaries
        """
        results = []
        for i, template in enumerate(query_templates):
            try:
                result = self.search(template, top_k, similarity_threshold)
                results.append(result)
                logger.debug(f"Processed query {i+1}/{len(query_templates)}")
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                results.append({
                    'query_template': template,
                    'results': [],
                    'total_results': 0,
                    'error': str(e)
                })
        
        return results
    
    def cluster_templates(
        self,
        clustering_threshold: float = 0.8,
        min_cluster_size: int = 2
    ) -> Dict[str, Any]:
        """
        Cluster templates based on similarity.
        
        Args:
            clustering_threshold: Similarity threshold for clustering
            min_cluster_size: Minimum size for a cluster
            
        Returns:
            Clustering results dictionary
        """
        if not self.template_metadata:
            logger.warning("No templates in index for clustering")
            return {'clusters': [], 'total_clusters': 0}
        
        # Extract all template vectors
        vectors = np.array([t['template_vector'] for t in self.template_metadata], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(vectors, vectors.T)
        
        # Apply clustering threshold
        adjacency_matrix = similarity_matrix >= clustering_threshold
        
        # Simple clustering using connected components
        clusters = self._connected_components_clustering(adjacency_matrix, min_cluster_size)
        
        # Format results
        cluster_results = []
        for i, cluster_indices in enumerate(clusters):
            cluster_templates = [self.template_metadata[idx] for idx in cluster_indices]
            cluster_results.append({
                'cluster_id': i,
                'size': len(cluster_templates),
                'templates': cluster_templates,
                'indices': cluster_indices
            })
        
        logger.info(f"Clustering complete: {len(cluster_results)} clusters found")
        return {
            'clusters': cluster_results,
            'total_clusters': len(cluster_results),
            'clustering_params': {
                'threshold': clustering_threshold,
                'min_cluster_size': min_cluster_size
            }
        }
    
    def _connected_components_clustering(
        self,
        adjacency_matrix: np.ndarray,
        min_cluster_size: int
    ) -> List[List[int]]:
        """Simple connected components clustering."""
        from scipy.sparse.csgraph import connected_components
        
        # Convert to sparse matrix
        from scipy.sparse import csr_matrix
        sparse_matrix = csr_matrix(adjacency_matrix)
        
        # Find connected components
        n_components, labels = connected_components(sparse_matrix, directed=False)
        
        # Group by component
        clusters = [[] for _ in range(n_components)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        # Filter by minimum size
        clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
        
        return clusters
    
    def save_index(self, output_path: Path) -> None:
        """Save index to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_path.with_suffix('.faiss')
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = output_path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'template_metadata': self.template_metadata,
                'index_params': {
                    'index_type': self.index_type,
                    'metric': self.metric,
                    'nlist': self.nlist,
                    'nprobe': self.nprobe,
                    'dimension': self.dimension
                }
            }, f)
        
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    @classmethod
    def load_index(
        cls,
        index_path: Path,
        metadata_path: Optional[Path] = None
    ) -> 'FundusFAISSIndex':
        """Load index from disk."""
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.pkl')
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            index_type=metadata['index_params']['index_type'],
            metric=metadata['index_params']['metric'],
            nlist=metadata['index_params']['nlist'],
            nprobe=metadata['index_params']['nprobe']
        )
        
        instance.index = index
        instance.template_metadata = metadata['template_metadata']
        instance.dimension = metadata['index_params']['dimension']
        instance.is_trained = True
        
        logger.info(f"Loaded index from {index_path}")
        return instance
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index."""
        if self.index is None:
            return {'status': 'not_created'}
        
        info = {
            'index_type': self.index_type,
            'metric': self.metric,
            'dimension': self.dimension,
            'total_templates': len(self.template_metadata),
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu
        }
        
        if hasattr(self.index, 'ntotal'):
            info['index_size'] = self.index.ntotal
        
        return info
