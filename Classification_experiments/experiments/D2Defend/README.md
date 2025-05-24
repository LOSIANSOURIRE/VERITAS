# D2Defend Model for Image Classification
=============================================
## Overview
This project implements a D2Defend-based model for image classification tasks. The model utilizes advanced denoising techniques, including bilateral filtering, short-time Fourier transform, and wavelet shrinkage, to improve its robustness against adversarial attacks.

## Architecture
The D2Defend model consists of the following components:
Bilateral Filtering Layer: Applies bilateral filtering to the input image to preserve edges and reduce noise.
Short-time Fourier Transform (STFT) Layer: Analyzes the texture of the input image using STFT.
Wavelet Shrinkage Layer: Applies wavelet shrinkage to denoise the texture component of the input image.
Inverse Short-time Fourier Transform (ISTFT) Layer: Reconstructs the denoised texture component using ISTFT.
Classification Layer: A multi-stage convolutional neural network (CNN) that classifies the input image into one of the predefined categories.

## Requirements
The following dependencies are required to run the project:
torch >= 1.8.0: PyTorch for model implementation and training.
torchvision >= 0.9.0: For image transformations and dataset loading.
scikit-learn >= 0.24.0: For performance evaluation and metrics.
pillow >= 8.0.0: For image processing.
pandas >= 1.1.0: For data manipulation (if needed).
PyWavelets>=1.1.0


## Usage
To train the D2Defend model, follow these steps:
Prepare your dataset by splitting it into training, validation, and testing sets.
Define the data transforms and create data loaders for each set.
Initialize the D2Defend model, optimizer, and criterion.
Train the model using the training data loader and evaluate its performance on the validation data loader.
Use early stopping to prevent overfitting and save the best-performing model weights.
Make inferences on the test dataset using the trained model.

The framework will provide:
- Forgery prediction (REAL or FAKE)
- Confidence score
- Detailed analysis metrics
- Visualization of artifact representations