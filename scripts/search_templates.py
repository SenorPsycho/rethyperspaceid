#!/usr/bin/env python3
"""
Search script for querying RetHyperspaceID biometric templates.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.logger import setup_logger
from preprocess.normalization import FundusNormalizer
from models.embedding_model import FundusEmbeddingModel
from privacy.cancellable_templates import CancellableTemplateGenerator
from index.faiss_index import FundusFAISSIndex


def load_and_process_image(image_path: Path, config: Config) -> np.ndarray:
    """Load and process a single image for search."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Normalize image
    normalizer = FundusNormalizer(
        target_size=tuple(config.get('preprocessing.target_size', [224, 224])),
        normalization=config.get('preprocessing.normalization', 'imagenet'),
        augmentation=False  # No augmentation for search
    )
    
    normalized = normalizer.normalize_image(image)
    return normalized


def main():
    """Main search execution."""
    parser = argparse.ArgumentParser(description="RetHyperspaceID Template Search")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--query-image", type=str, required=True, help="Path to query fundus image")
    parser.add_argument("--index-path", type=str, help="Path to FAISS index")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Minimum similarity threshold")
    parser.add_argument("--output-file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    logger = setup_logger(
        level=config.get('logging.level', 'INFO'),
        log_file=False
    )
    
    logger.info("Starting RetHyperspaceID template search")
    
    # Determine index path
    if args.index_path:
        index_path = Path(args.index_path)
    else:
        index_path = Path(config.get('data.artifacts_dir', 'artifacts')) / "faiss_index"
    
    # Check if index exists
    if not index_path.with_suffix('.faiss').exists():
        logger.error(f"FAISS index not found at {index_path}")
        return 1
    
    # Load query image
    query_image_path = Path(args.query_image)
    if not query_image_path.exists():
        logger.error(f"Query image not found: {query_image_path}")
        return 1
    
    try:
        logger.info(f"Processing query image: {query_image_path}")
        normalized_image = load_and_process_image(query_image_path, config)
    except Exception as e:
        logger.error(f"Failed to process query image: {e}")
        return 1
    
    # Load embedding model
    try:
        model_path = Path(config.get('data.artifacts_dir', 'artifacts')) / "embedding_model.pth"
        if model_path.exists():
            embedding_model = FundusEmbeddingModel.load_model(model_path)
            logger.info("Loaded existing embedding model")
        else:
            # Create new model
            embedding_model = FundusEmbeddingModel(
                architecture=config.get('model.architecture', 'efficientnet_b0'),
                embedding_dim=config.get('model.embedding_dim', 1280),
                pretrained=config.get('model.pretrained', True),
                device=config.get('model.device', 'auto')
            )
            logger.info("Created new embedding model")
    except Exception as e:
        logger.error(f"Failed to load/create embedding model: {e}")
        return 1
    
    # Extract embedding
    try:
        logger.info("Extracting embedding from query image")
        embedding = embedding_model.extract_embedding(normalized_image)
        logger.info(f"Extracted embedding with shape: {embedding.shape}")
    except Exception as e:
        logger.error(f"Failed to extract embedding: {e}")
        return 1
    
    # Generate template
    try:
        logger.info("Generating cancellable template")
        template_generator = CancellableTemplateGenerator(
            template_dim=config.get('privacy.template_dim', 512),
            random_projection_seed=config.get('privacy.random_projection_seed', 42),
            sign_code_bits=config.get('privacy.sign_code_bits', 256),
            cancellable=config.get('privacy.cancellable', True)
        )
        
        template = template_generator.generate_template(embedding, "query")
        logger.info("Generated query template")
    except Exception as e:
        logger.error(f"Failed to generate template: {e}")
        return 1
    
    # Load FAISS index
    try:
        logger.info("Loading FAISS index")
        faiss_index = FundusFAISSIndex.load_index(index_path)
        logger.info(f"Loaded index with {faiss_index.get_index_info()['total_templates']} templates")
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return 1
    
    # Search for similar templates
    try:
        logger.info("Searching for similar templates")
        search_results = faiss_index.search(
            template,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold
        )
        
        logger.info(f"Search complete: {search_results['total_results']} results found")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    # Format and display results
    print(f"\nSearch Results for: {query_image_path.name}")
    print(f"Total results: {search_results['total_results']}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print("-" * 80)
    
    if search_results['total_results'] > 0:
        print(f"{'Rank':<5} {'Similarity':<12} {'User ID':<15} {'Distance':<12}")
        print("-" * 80)
        
        for result in search_results['results']:
            print(f"{result['rank']:<5} {result['similarity']:<12.4f} {result['template']['user_id']:<15} {result['distance']:<12.4f}")
    else:
        print("No similar templates found above the similarity threshold.")
    
    # Save results if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for saving
        save_results = {
            'query_image': str(query_image_path),
            'search_timestamp': str(np.datetime64('now')),
            'search_params': search_results['search_params'],
            'total_results': search_results['total_results'],
            'results': []
        }
        
        for result in search_results['results']:
            save_results['results'].append({
                'rank': result['rank'],
                'similarity': result['similarity'],
                'distance': result['distance'],
                'user_id': result['template']['user_id'],
                'template_metadata': {
                    'template_dim': result['template']['template_dim'],
                    'sign_code_bits': result['template']['sign_code_bits'],
                    'cancellable': result['template']['cancellable']
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        logger.info(f"Search results saved to {output_path}")
    
    logger.info("RetHyperspaceID template search completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
