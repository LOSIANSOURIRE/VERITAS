# Vision Transformer (ViT) and DEFL Model for AI-Generated Image Detection
## Overview
This model implements a Vision Transformer (ViT) combined with a Directional Enhanced Feature Learning (DEFL) model for AI-generated image detection. The model is designed to classify images as real or AI-generated (fake) based on features extracted through DEFL and processed by a ViT model. It integrates directional convolutions and standard convolutions for rich feature learning, and uses a combined BCE loss and contrastive loss for effective training. The model uses PyTorch for model implementation and training, and timm for Vision Transformer (ViT) model initialization. Image preprocessing includes FFT and LBP features, along with image augmentation for improved generalization.

## Architecture
### DEFL Model:
Directional Convolutional Blocks (DCBs): Apply composite directional filters to capture high-frequency directional features.

Standard Convolutional Blocks (SCBs): Apply standard convolutions for general feature extraction.

Vision Transformer (ViT):
Pre-trained ViT: The base model is a pre-trained ViT model with a custom classification head for binary classification (real or fake).
CLS Token: The embedding from the CLS token of the ViT model is used for final classification.

### Combined Loss:
A Combined Loss Function that integrates Binary Cross-Entropy (BCE) and Contrastive Loss is used for model training.

## Dataset and Training
The model is trained on the custom_dataset/train directory and evaluated on the custom_dataset/test directory.
The model extracts features using DEFL, augments them with FFT and LBP features, and then feeds them to the ViT model.
During training, cross-entropy loss and contrastive loss are used to optimize the model. The Adam optimizer is used for training.
After training, the model weights are saved as defl_weights.pth and vit_weights.pth.

## Inputs and Outputs
Input: Images from dataset-mix/train for training and dataset-mix/test for testing. Additionally, predictions are made on the dataset located in test_dataset/perturbed_images_32.

Outputs:
Trained model weights saved as defl_weights.pth and vit_weights.pth.
A CSV file (predictions.csv) containing the predicted classes (0 for fake, 1 for real) for the perturbed images in the final test dataset.
Model Usage


## Requirements
The following dependencies are required to run the project:

torch >= 1.8.0: PyTorch for model implementation and training.
torchvision >= 0.9.0: For image transformations and dataset loading.
scikit-learn >= 0.24.0: For performance evaluation and metrics.
timm >= 0.4.12: For pre-trained Vision Transformer models.
scipy >= 1.5.4: For convolution operations and numerical stability.
scikit-image >= 0.18.3: For Local Binary Pattern (LBP) feature extraction.
opencv-python-headless >= 4.5.1: For image processing.
pandas >= 1.1.0: For CSV file handling and data manipulation.

## Instructions
To run the code, ensure you first install all requirements and then unzip the datasets which are present in the folder containing the .ipynb file. Then the cells can be run sequentially to see the desired outputs. 


