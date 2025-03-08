# AuraSR: GAN-based Super-Resolution Implementation

## Overview
AuraSR is a GAN-based super-resolution model designed to enhance the quality of images, specifically optimized for real-world applications. This implementation is heavily modified from the unofficial `lucidrains/gigagan-pytorch` repository and utilizes techniques from the GigaGAN model. The project is aimed at improving image resolution using advanced neural network architectures.

## Features
- **Super-Resolution**: Enhances low-resolution images to high-resolution outputs.
- **Adaptive Convolution**: Utilizes adaptive convolutional layers to improve feature extraction.
- **Attention Mechanisms**: Implements attention layers for better context understanding in images.
- **Modular Design**: Composed of various modular components like blocks, transformers, and attention layers for flexibility and scalability.

## Requirements
To run this project, you'll need to install the following dependencies. You can do this by executing:
```bash
pip install -r Requirements_AuraSR_CIFAR10.txt
```

### Dependencies Include:
- PyTorch
- torchvision
- einops
- PIL
- OpenCV
- Other libraries specified in the requirements file.

## Installation
1. Clone the repository or download the files.
2. Navigate to the project directory.
3. Install the required packages as mentioned above.

## Architecture
The architecture includes several key components:
- **AdaptiveConv2DMod**: A convolutional layer with adaptive kernels.
- **Attention Mechanisms**: Implemented through classes like `Attention`, `LinearAttention`, and `Transformer`.
- **Upsampling Techniques**: Utilizes methods like nearest neighbor upsampling to enhance image resolution.


```

