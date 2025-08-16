#!/usr/bin/env python3
"""
Test script to verify RetHyperspaceID installation and components.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.config import Config
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Config: {e}")
        return False
    
    try:
        from utils.logger import setup_logger
        print("✓ Logger module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Logger: {e}")
        return False
    
    try:
        from preprocess.quality_assessment import FundusQualityAssessor
        print("✓ Quality assessment module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Quality assessment: {e}")
        return False
    
    try:
        from preprocess.normalization import FundusNormalizer
        print("✓ Normalization module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Normalization: {e}")
        return False
    
    try:
        from models.embedding_model import FundusEmbeddingModel
        print("✓ Embedding model module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Embedding model: {e}")
        return False
    
    try:
        from privacy.cancellable_templates import CancellableTemplateGenerator
        print("✓ Privacy module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Privacy module: {e}")
        return False
    
    try:
        from index.faiss_index import FundusFAISSIndex
        print("✓ FAISS index module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FAISS index: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False
    
    try:
        import timm
        print(f"✓ Timm {timm.__version__} available")
    except ImportError as e:
        print(f"✗ Timm not available: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} available")
    except ImportError as e:
        print(f"✗ OpenCV not available: {e}")
        return False
    
    try:
        import faiss
        print(f"✓ FAISS available")
    except ImportError as e:
        print(f"✗ FAISS not available: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} available")
    except ImportError as e:
        print(f"✗ NumPy not available: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} available")
    except ImportError as e:
        print(f"✗ Pandas not available: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} available")
    except ImportError as e:
        print(f"✗ Scikit-learn not available: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from utils.config import Config
        config = Config()
        print("✓ Configuration loaded successfully")
        
        # Test some config values
        target_size = config.get('preprocessing.target_size', [224, 224])
        print(f"  Target size: {target_size}")
        
        model_arch = config.get('model.architecture', 'efficientnet_b0')
        print(f"  Model architecture: {model_arch}")
        
        template_dim = config.get('privacy.template_dim', 512)
        print(f"  Template dimension: {template_dim}")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    return True


def test_components():
    """Test component initialization."""
    print("\nTesting component initialization...")
    
    try:
        # Test quality assessor
        from preprocess.quality_assessment import FundusQualityAssessor
        qa = FundusQualityAssessor(quality_threshold=0.8)
        print("✓ Quality assessor initialized")
        
        # Test normalizer
        from preprocess.normalization import FundusNormalizer
        normalizer = FundusNormalizer()
        print("✓ Normalizer initialized")
        
        # Test template generator
        from privacy.cancellable_templates import CancellableTemplateGenerator
        template_gen = CancellableTemplateGenerator()
        print("✓ Template generator initialized")
        
        # Test FAISS index
        from index.faiss_index import FundusFAISSIndex
        faiss_idx = FundusFAISSIndex()
        print("✓ FAISS index initialized")
        
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        return False
    
    return True


def test_embedding_model():
    """Test embedding model (this may take a moment)."""
    print("\nTesting embedding model...")
    
    try:
        from models.embedding_model import FundusEmbeddingModel
        
        # Create model
        model = FundusEmbeddingModel(device='cpu')  # Use CPU for testing
        print("✓ Embedding model created")
        
        # Test with dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        embedding = model.extract_embedding(dummy_input)
        print(f"✓ Embedding extracted, shape: {embedding.shape}")
        
        # Test model info
        info = model.get_model_info()
        print(f"  Model info: {info['architecture']}, {info['embedding_dim']} dims")
        
    except Exception as e:
        print(f"✗ Embedding model test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("RetHyperspaceID Installation Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("Components", test_components),
        ("Embedding Model", test_embedding_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! RetHyperspaceID is ready to use.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
