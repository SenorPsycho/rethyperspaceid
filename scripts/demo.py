#!/usr/bin/env python3
"""
Demo script for RetHyperspaceID showing the complete pipeline.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.logger import setup_logger
from preprocess.quality_assessment import FundusQualityAssessor
from preprocess.normalization import FundusNormalizer
from models.embedding_model import FundusEmbeddingModel
from privacy.cancellable_templates import CancellableTemplateGenerator
from index.faiss_index import FundusFAISSIndex


def create_demo_images(num_images: int = 10, size: tuple = (224, 224)) -> list:
    """Create synthetic demo fundus images for demonstration."""
    print(f"Creating {num_images} synthetic demo images...")
    
    images = []
    for i in range(num_images):
        # Create synthetic fundus-like image
        img = np.random.randint(50, 200, (*size, 3), dtype=np.uint8)
        
        # Add some structure to make it look more realistic
        # Add circular fundus boundary
        center = (size[1] // 2, size[0] // 2)
        radius = min(size) // 2 - 20
        
        for y in range(size[0]):
            for x in range(size[1]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist > radius:
                    img[y, x] = [20, 20, 20]  # Dark border
        
        # Add some "vessels" (random lines)
        for _ in range(5):
            start = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
            end = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
            cv2.line(img, start, end, (100, 100, 100), 2)
        
        images.append(img)
    
    return images


def save_demo_images(images: list, output_dir: Path):
    """Save demo images to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for i, img in enumerate(images):
        path = output_dir / f"demo_fundus_{i:03d}.png"
        cv2.imwrite(str(path), img)
        image_paths.append(path)
    
    print(f"Saved {len(images)} demo images to {output_dir}")
    return image_paths


def run_demo_pipeline():
    """Run the complete demo pipeline."""
    print("RetHyperspaceID Demo Pipeline")
    print("=" * 50)
    
    # Setup
    config = Config()
    logger = setup_logger(level="INFO", log_file=False)
    
    # Create demo data directory
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # Step 1: Create demo images
    print("\n1. Creating demo fundus images...")
    demo_images = create_demo_images(num_images=10)
    image_paths = save_demo_images(demo_images, demo_dir / "raw")
    
    # Step 2: Quality Assessment
    print("\n2. Running quality assessment...")
    quality_assessor = FundusQualityAssessor(quality_threshold=0.5)  # Lower threshold for demo
    
    # Convert images to paths for quality assessment
    passed_images, failed_images = quality_assessor.filter_images(image_paths)
    print(f"   Quality assessment: {len(passed_images)} passed, {len(failed_images)} failed")
    
    # Step 3: Image Normalization
    print("\n3. Normalizing images...")
    normalizer = FundusNormalizer(
        target_size=(224, 224),
        normalization="imagenet",
        augmentation=False
    )
    
    # Process passed images
    processed_images = []
    for img_path in passed_images:
        img = cv2.imread(str(img_path))
        normalized = normalizer.normalize_image(img)
        processed_images.append(normalized)
    
    print(f"   Normalized {len(processed_images)} images")
    
    # Step 4: Embedding Extraction
    print("\n4. Extracting embeddings...")
    embedding_model = FundusEmbeddingModel(
        architecture="efficientnet_b0",
        device="cpu"  # Use CPU for demo
    )
    
    # Convert to numpy array
    images_array = np.array(processed_images)
    embeddings = embedding_model.extract_embeddings_batch(images_array, batch_size=4)
    print(f"   Extracted embeddings: {embeddings.shape}")
    
    # Step 5: Template Generation
    print("\n5. Generating cancellable templates...")
    template_generator = CancellableTemplateGenerator(
        template_dim=512,
        random_projection_seed=42,
        sign_code_bits=256,
        cancellable=True
    )
    
    user_ids = [f"demo_user_{i:03d}" for i in range(len(embeddings))]
    templates = template_generator.generate_templates_batch(embeddings, user_ids)
    print(f"   Generated {len(templates)} templates")
    
    # Step 6: Create FAISS Index
    print("\n6. Creating FAISS index...")
    faiss_index = FundusFAISSIndex(
        index_type="faiss",
        metric="cosine",
        use_gpu=False
    )
    
    faiss_index.create_index(512)
    faiss_index.add_templates(templates)
    print(f"   Created index with {faiss_index.get_index_info()['total_templates']} templates")
    
    # Step 7: Search Demo
    print("\n7. Running search demo...")
    
    # Use first template as query
    query_template = templates[0]
    search_results = faiss_index.search(
        query_template,
        top_k=5,
        similarity_threshold=0.5
    )
    
    print(f"   Search results for {query_template['user_id']}:")
    for result in search_results['results']:
        print(f"     Rank {result['rank']}: {result['template']['user_id']} "
              f"(similarity: {result['similarity']:.3f})")
    
    # Step 8: Clustering Demo
    print("\n8. Running clustering demo...")
    clusters = faiss_index.cluster_templates(
        clustering_threshold=0.7,
        min_cluster_size=2
    )
    
    print(f"   Found {clusters['total_clusters']} clusters:")
    for cluster in clusters['clusters']:
        user_ids = [t['user_id'] for t in cluster['templates']]
        print(f"     Cluster {cluster['cluster_id']}: {len(user_ids)} templates - {user_ids}")
    
    # Step 9: Template Cancellation Demo
    print("\n9. Demonstrating template cancellation...")
    
    # Generate new template with different seed
    new_template = template_generator.regenerate_template(
        embeddings[0],
        new_seed=12345,
        user_id="demo_user_000_cancelled"
    )
    
    # Search with new template
    new_search_results = faiss_index.search(
        new_template,
        top_k=3,
        similarity_threshold=0.5
    )
    
    print(f"   New template search results:")
    for result in new_search_results['results']:
        print(f"     Rank {result['rank']}: {result['template']['user_id']} "
              f"(similarity: {result['similarity']:.3f})")
    
    # Step 10: Save Results
    print("\n10. Saving demo results...")
    
    # Save templates
    templates_path = demo_dir / "templates.pkl"
    template_generator.save_templates(templates, templates_path)
    
    # Save index
    index_path = demo_dir / "faiss_index"
    faiss_index.save_index(index_path)
    
    # Save embeddings
    embeddings_path = demo_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    print(f"   Results saved to {demo_dir}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"Check the {demo_dir} directory for generated files.")
    
    return True


def main():
    """Main demo execution."""
    try:
        success = run_demo_pipeline()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
