# RetHyperspaceID

A local, privacy-preserving retina biometric identification prototype that processes fundus images to create cancellable biometric templates for secure identification.

## Overview

RetHyperspaceID implements a complete pipeline for retina biometric identification with strong privacy guarantees:

1. Quality Assessment & Normalization: Filters and preprocesses fundus images
2. Feature Extraction: Uses CNN models (EfficientNet) to extract embeddings
3. Privacy Protection: Generates cancellable templates using random projection and sign codes
4. Efficient Search: FAISS-based similarity search and clustering
5. Local Processing: Everything runs locally - no cloud services required

## Key Features

- Privacy-Preserving: No raw biometrics stored, only cancellable templates
- Fully Local: Complete offline processing pipeline
- High Quality: Advanced quality assessment and image normalization
- Fast Search: FAISS-based similarity search with GPU acceleration support
- Cancellable: Templates can be regenerated with new random seeds
- Scalable: Handles large datasets (25k+ images) efficiently

