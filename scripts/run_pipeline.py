#!/usr/bin/env python3
"""
Main pipeline script for RetHyperspaceID retina biometric system.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.logger import setup_logger
from preprocess.quality_assessment import FundusQualityAssessor
from preprocess.normalization import FundusNormalizer
from models.embedding_model import FundusEmbeddingModel
from privacy.cancellable_templates import CancellableTemplateGenerator
from index.faiss_index import FundusFAISSIndex


def setup_seeds(seed: int = 42) -> None:
    """Setup deterministic random seeds."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_images_from_directory(image_dir: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> list:
    """Load image paths from directory."""
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    
    return sorted(image_paths)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="RetHyperspaceID Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--input-dir", type=str, help="Input directory with raw images")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed data")
    parser.add_argument("--skip-qa", action="store_true", help="Skip quality assessment")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip image normalization")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--skip-templates", action="store_true", help="Skip template generation")
    parser.add_argument("--skip-index", action="store_true", help="Skip index creation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    setup_seeds(args.seed)
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    logger = setup_logger(
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.save_logs', True) and "pipeline.log",
        log_dir=config.get('logging.log_dir', 'logs')
    )
    
    logger.info("Starting RetHyperspaceID pipeline")
    
    # Determine input and output directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(config.get('data.raw_dir', 'data/raw'))
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('data.processed_dir', 'data/processed'))
    
    artifacts_dir = Path(config.get('data.artifacts_dir', 'artifacts'))
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load images
    logger.info(f"Loading images from {input_dir}")
    image_paths = load_images_from_directory(input_dir)
    
    if not image_paths:
        logger.error(f"No images found in {input_dir}")
        return 1
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Step 2: Quality Assessment
    if not args.skip_qa:
        logger.info("Starting quality assessment")
        quality_assessor = FundusQualityAssessor(
            quality_threshold=config.get('preprocessing.quality_threshold', 0.8)
        )
        
        passed_images, failed_images = quality_assessor.filter_images(image_paths)
        
        logger.info(f"Quality assessment complete: {len(passed_images)} passed, {len(failed_images)} failed")
        
        # Save quality assessment results
        qa_results = {
            'total_images': len(image_paths),
            'passed_images': [str(p) for p in passed_images],
            'failed_images': [str(p) for p in failed_images],
            'pass_rate': len(passed_images) / len(image_paths)
        }
        
        import json
        with open(artifacts_dir / "qa_results.json", 'w') as f:
            json.dump(qa_results, f, indent=2)
    else:
        logger.info("Skipping quality assessment")
        passed_images = image_paths
    
    # Step 3: Image Normalization
    if not args.skip_normalization and passed_images:
        logger.info("Starting image normalization")
        normalizer = FundusNormalizer(
            target_size=tuple(config.get('preprocessing.target_size', [224, 224])),
            normalization=config.get('preprocessing.normalization', 'imagenet'),
            augmentation=config.get('preprocessing.augmentation', False)
        )
        
        norm_stats = normalizer.batch_normalize(
            passed_images,
            output_dir,
            quality_passed=passed_images if not args.skip_qa else None
        )
        
        logger.info(f"Normalization complete: {norm_stats['processed']} processed")
        
        # Update passed images to normalized images
        normalized_images = list(output_dir.glob("norm_*.png"))
        passed_images = normalized_images
    else:
        logger.info("Skipping image normalization")
    
    # Step 4: Embedding Extraction
    if not args.skip_embedding and passed_images:
        logger.info("Starting embedding extraction")
        
        # Load normalized images
        images = []
        for img_path in tqdm(passed_images, desc="Loading images"):
            try:
                # Load normalized image (RGB format)
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Normalize to [0, 1] and apply ImageNet normalization
                    img_norm = img_rgb.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_norm = (img_norm - mean) / std
                    images.append(img_norm)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        if not images:
            logger.error("No images loaded for embedding extraction")
            return 1
        
        images = np.array(images)
        logger.info(f"Loaded {len(images)} images for embedding extraction")
        
        # Initialize embedding model
        embedding_model = FundusEmbeddingModel(
            architecture=config.get('model.architecture', 'efficientnet_b0'),
            embedding_dim=config.get('model.embedding_dim', 1280),
            pretrained=config.get('model.pretrained', True),
            device=config.get('model.device', 'auto')
        )
        
        # Extract embeddings
        batch_size = config.get('training.batch_size', 32)
        embeddings = embedding_model.extract_embeddings_batch(
            images,
            batch_size=batch_size,
            show_progress=True
        )
        
        # Save embeddings
        embeddings_path = artifacts_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save model
        model_path = artifacts_dir / "embedding_model.pth"
        embedding_model.save_model(model_path)
        
    else:
        logger.info("Skipping embedding extraction")
        # Try to load existing embeddings
        embeddings_path = artifacts_dir / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            logger.info(f"Loaded existing embeddings: {embeddings.shape}")
        else:
            logger.error("No embeddings available and embedding extraction skipped")
            return 1
    
    # Step 5: Template Generation
    if not args.skip_templates:
        logger.info("Starting template generation")
        
        template_generator = CancellableTemplateGenerator(
            template_dim=config.get('privacy.template_dim', 512),
            random_projection_seed=config.get('privacy.random_projection_seed', 42),
            sign_code_bits=config.get('privacy.sign_code_bits', 256),
            cancellable=config.get('privacy.cancellable', True)
        )
        
        # Generate user IDs
        user_ids = [f"user_{i:06d}" for i in range(len(embeddings))]
        
        # Generate templates
        templates = template_generator.generate_templates_batch(embeddings, user_ids)
        
        # Save templates
        templates_path = artifacts_dir / "templates.pkl"
        template_generator.save_templates(templates, templates_path)
        logger.info(f"Generated and saved {len(templates)} templates")
        
    else:
        logger.info("Skipping template generation")
        # Try to load existing templates
        templates_path = artifacts_dir / "templates.pkl"
        if templates_path.exists():
            templates = CancellableTemplateGenerator.load_templates(templates_path)
            logger.info(f"Loaded existing templates: {len(templates)}")
        else:
            logger.error("No templates available and template generation skipped")
            return 1
    
    # Step 6: Index Creation
    if not args.skip_index:
        logger.info("Starting index creation")
        
        faiss_index = FundusFAISSIndex(
            index_type=config.get('index.type', 'faiss'),
            metric=config.get('index.metric', 'cosine'),
            nlist=config.get('index.nlist', 100),
            nprobe=config.get('index.nprobe', 10),
            use_gpu=config.get('index.use_gpu', False)
        )
        
        # Create index
        template_dim = config.get('privacy.template_dim', 512)
        faiss_index.create_index(template_dim)
        
        # Add templates
        faiss_index.add_templates(templates)
        
        # Save index
        index_path = artifacts_dir / "faiss_index"
        faiss_index.save_index(index_path)
        logger.info(f"Created and saved FAISS index")
        
        # Perform clustering
        clustering_threshold = config.get('search.clustering_threshold', 0.8)
        clusters = faiss_index.cluster_templates(clustering_threshold)
        logger.info(f"Clustering complete: {clusters['total_clusters']} clusters found")
        
        # Save clustering results
        import json
        cluster_results = {
            'total_clusters': clusters['total_clusters'],
            'clustering_threshold': clustering_threshold,
            'clusters': [
                {
                    'cluster_id': c['cluster_id'],
                    'size': c['size'],
                    'user_ids': [t['user_id'] for t in c['templates']]
                }
                for c in clusters['clusters']
            ]
        }
        
        with open(artifacts_dir / "clustering_results.json", 'w') as f:
            json.dump(cluster_results, f, indent=2)
        
    else:
        logger.info("Skipping index creation")
    
    logger.info("RetHyperspaceID pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
