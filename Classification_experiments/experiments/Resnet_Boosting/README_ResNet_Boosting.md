# Siva Sequential Boosting Model

## Overview

This repository contains a PyTorch implementation of a sequential boosting model that utilizes adversarial attacks for image classification tasks. The model is built on a modified ResNet-18 architecture and is designed to handle binary classification problems. The code includes functionalities for training the model, applying data augmentation, and evaluating performance under normal and adversarial conditions.

## Features

- **Model Architecture**: A modified ResNet-18 tailored for binary classification.
- **Data Handling**: Utilizes PyTorch's `DataLoader` for efficient data loading and augmentation.
- **Adversarial Attack Implementation**: Implements Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) for testing model robustness.
- **Performance Metrics**: Evaluates model performance using accuracy, F1 score, precision, recall, and classification reports.

## Installation


You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Data**: Organize your dataset into `train` and `test` directories with subdirectories for each class.

2. **Run the Model**:
   - Load the Jupyter Notebook `Resnet
   - resnet_sequential_boosting_like.ipynb` in Google Colab or your local Jupyter environment.
   - Execute the cells sequentially to train the model and evaluate its performance.

## Code Structure

### Key Components

- **Imports**: Essential libraries are imported at the beginning of the notebook.

- **Transforms**: Data normalization and tensor conversion are handled using `torchvision.transforms`.


### Training and Evaluation

The training process includes logging of loss and accuracy metrics for both training and validation phases. Adversarial attacks are applied during evaluation to assess model robustness.
