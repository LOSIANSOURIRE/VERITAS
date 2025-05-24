# Ensemble Model for Image Classification

## Overview

This project implements an advanced ensemble model for classifying images as either "real" or "fake" using multiple state-of-the-art deep learning architectures. By leveraging the strengths of different neural network models, we aim to improve overall prediction accuracy and robustness.

## Features

- Multiple Model Architectures:
  - EfficientNet V2-S
  - Vision Transformer (ViT)
  - ResNet-18

- Advanced Ensemble Prediction Techniques
- Comprehensive Data Preprocessing
- Performance Metrics Tracking

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Minimum 16GB RAM
- At least 50GB free disk space

### Required Libraries
- PyTorch
- torchvision
- timm
- scikit-learn
- pandas
- numpy
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ensemble-image-classification.git
cd ensemble-image-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Directory Structure
```
dataset/
├── Folder
│   ├── image1.png
│   ├── image2.png
│   └── ...

```

### Data Organization
- unzip the model weights zip.
- unzip the dataset zip
- set path in test loader and model weights loader.
- Supported image formats: .jpg, .png, .jpeg

## Training the Models

### Configuration
Edit the configuration parameters in `train.py`:
- `DATASET_PATH`: Path to your dataset
- `BATCH_SIZE`: Number of images per batch
- `NUM_EPOCHS`: Training epochs
- `LEARNING_RATE`: Optimization learning rate



## Ensemble Prediction

The ensemble model combines predictions from multiple architectures:
- Weighted averaging of model predictions
- Adaptive weight assignment based on individual model performance

## Output

### Prediction Formats
1. CSV: Image paths with predicted probabilities
2. JSON: Structured prediction results

### Example JSON Output
```json
[
  { "index": 1, "prediction": "real", "confidence": 0.92 },
  { "index": 2, "prediction": "fake", "confidence": 0.65 }
]
```

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score


### Model Weights
Adjust individual model weights in `ensemble_weights.zip` to fine-tune ensemble performance.


## Troubleshooting

- Ensure GPU drivers are up to date
- Check CUDA compatibility
- Verify library versions
- Monitor GPU memory usage




