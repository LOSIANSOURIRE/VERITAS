# ResNeXt Model for Image Classification
This contains an implementation of a ResNeXt-based model for AI-generated image detection. The model utilizes a group convolution-based architecture that is optimized for performance in AI-generated image detection. 

## Overview
The project focuses on building and training a ResNeXt-based deep learning model for AI-generated image detection. It uses PyTorch for model implementation and training, and torchvision for image transformations and dataset handling.

The model is trained using image datasets, and the directory structure supports both training and testing phases. The architecture leverages the ResNeXt design to achieve better performance on large-scale image classification tasks.

## Architecture
The architecture implemented in this project follows the ResNeXt model, specifically the **ResNeXt29_2x64d** variation. Here’s an overview of how it works:

1. **Block Structure**: The core of the ResNeXt model is based on a **bottleneck block** with **group convolutions**. The key features of each block are:
    - 1x1 convolution (expansion)
    - 3x3 group convolution (cardinality-based)
    - 1x1 convolution (compression)
    - Shortcut connections (for residual learning)

2. **Model Design**:
    - The model uses a **customizable number of blocks**, **cardinality**, and **bottleneck width**.
    - It is optimized for binary classification, though it can be adapted for multi-class classification as well.
    - The final layer is a fully connected layer that outputs the class probabilities.

3. **Training Strategy**:
    - The model is trained using **cross-entropy loss** and optimized using the **Adam optimizer**.
    - Transformations such as random horizontal flips and normalization are applied to the input images to augment the training data and improve generalization.
    
## Inputs and Outputs
The model is trained on data from **custom_dataset/train** and the saved weights are used to run validation tests on **custom_dataset/test**
This is followed by a prediction by the model on the final dataset **test_dataset/perturbed_images_32**

The model will output the weights of the ResNext architecture post training for 20 epochs in **resnext_weights.pth**
It will also output a CSV file, **final_predictions.csv** which contains the predicted classes (0 for FAKE, 1 for REAL) on the final dataset. This can be used for inference. 

## Requirements
The following dependencies are required to run the project:
- `torch >= 1.8.0`: PyTorch for model implementation and training.
- `torchvision >= 0.9.0`: For image transformations and dataset loading.
- `scikit-learn >= 0.24.0`: For performance evaluation and metrics.
- `pillow >= 8.0.0`: For image processing.
- `pandas >= 1.1.0`: For data manipulation (if needed).

## Instructions
To run the code, ensure you first install all requirements and then unzip the datasets which are present in the folder containing the .ipynb file. Then the cells can be run sequentially to see the desired outputs. 





