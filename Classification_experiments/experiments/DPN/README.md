# Image Classification with Dual Path Network (DPN)

## Overview

The **DPN_Final_Notebook.ipynb** is a Jupyter notebook designed for training, validating, and testing a Dual Path Network (DPN) model on a custom dataset. The primary objective is to predict whether images belong to the "Real" or "Fake" class using advanced image processing techniques, including Local Binary Patterns (LBP) and Gaussian filtering.

## Requirements

This notebook is built using Python 3.10.14 and requires the following libraries:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `skimage`
- `timm`
- `tqdm`
- Other dependencies as specified in the notebook

Make sure to install these packages in your environment before running the notebook.

## Features

### 1. Data Preprocessing
- Custom dataset class that applies LBP preprocessing
- Resizing images and converting them to RGB format
- Implementation of FFT (Fast Fourier Transform) alongside LBP for enhanced feature extraction

### 2. Model Architecture
- Utilizes a DPN architecture which combines features from multiple paths to improve prediction accuracy

### 3. Training and Evaluation
- Configurable parameters such as batch size, number of epochs, and learning rate
- Implements training loops with validation metrics including precision, recall, and F1-score

### 4. Data Augmentation
- Incorporates various transformations such as random horizontal/vertical flips and affine transformations to enhance the robustness of the model

### 5. Custom Transformations
- Classes for adding Gaussian noise and applying double-layer Gaussian filters to augment training data

## Usage Instructions

### 1. Setup Environment
- Ensure that you have a compatible Python environment with all required libraries installed.

### 2. Load Dataset
- Unzip the custom dataset, extract and save it in a path
- Under the Data Preparation Section, define the paths for train and test datasets from the extracted custom dataset
- Inside the Model Testing on New Dataset Section, under the Load Test Dataset sub section, define the path for the new dataset on which you need the model to provide predictions

### 3. Run Notebook
- Open the notebook in Jupyter or Google Colab, execute each cell sequentially, and monitor the training process through printed outputs and metrics.

### 4. Modify Parameters
- Feel free to adjust hyperparameters such as `batch_size`, `num_epochs`, and augmentation techniques based on your dataset characteristics and computational resources.

### 5. Evaluate Model
- After training, evaluate the model's performance on a test set using the provided evaluation metrics.

### 6. Testing the Model on New or Custom Dataset
- To test the model on a new dataset, ensure that the test images are placed directly inside the specified test directory (`test_dir`), with no subfolders. The `CustomTestDataset` class will load the images directly from the folder and apply the necessary transformations.
- **test_dir**: Provide the path to the folder containing the test images
- The images should be placed directly inside the folder, with **no subfolders**.
- The `CustomTestDataset` will load and transform the images as per the defined `transform`.
- The `DataLoader` will help in batching the data for testing.

## Conclusion

This notebook serves as a comprehensive guide for implementing a DPN model on image datasets, providing essential preprocessing steps, model training, and evaluation methodologies. Users can adapt it further based on specific project needs or datasets.