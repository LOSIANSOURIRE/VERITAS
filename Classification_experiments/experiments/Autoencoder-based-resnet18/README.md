# Autoencoder Model README

## Overview
The **Autoencoder** model is a deep learning architecture designed for image reconstruction and classification tasks. This implementation utilizes a U-Net based autoencoder structure combined with a ResNet-18 classifier, allowing it to effectively handle perturbed and unperturbed images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
  - [AutoEncoderPyTorch](#autoencoderpytorch)
  - [ResNet18](#resnet18)
  - [Autores_pipeline](#autores_pipeline)
- [Dataset Preparation](#dataset-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run the Autoencoder model, ensure you have Python and the following libraries installed:
```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn keras
```

## Usage
To use the Autoencoder model, you can import the necessary classes and instantiate the model as follows:

```python
from autoencoder import AutoEncoderPyTorch, ResNet18, Autores_pipeline

input_shape = (32, 32)  # Example input shape
model = Autores_pipeline(input_shape=input_shape, num_classes=2)
```

### Example of Forward Pass
```python
output = model(input_tensor)  # input_tensor should be a tensor of appropriate shape
```

## Model Architecture

### AutoEncoderPyTorch
The `AutoEncoderPyTorch` class implements a U-Net style architecture for encoding and decoding images. It consists of convolutional layers for feature extraction and transposed convolutional layers for reconstruction.

```python
class AutoEncoderPyTorch(nn.Module):
    def __init__(self, input_shape):
        # Initialize encoder and decoder layers
```

### ResNet18
The `ResNet18` class is a wrapper around the pre-trained ResNet-18 model used for classification tasks. It modifies the final layers to adapt to the specific number of output classes.
Please change the path of the **ResNet20** model which has same name as that present in the code and that path is present in this current folder only

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        # Initialize ResNet-18 with modified output layers
```

### Autores_pipeline
The `Autores_pipeline` class combines the autoencoder and classifier into a single pipeline. It processes input images through the autoencoder to generate clean images, which are then classified using the ResNet model.
Please change the path of the **Autores_pipeline** model which has same name as that present in the code and that path is present in this current folder only

```python
class Autores_pipeline(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        # Initialize autoencoder and classifier components
```

## Dataset Preparation
To prepare your dataset, ensure that you have unzipped the **autoencoder_x_label.zip** and **autoencoder_y_label.zip** files.Ensure you have two directories: one for **autoencoder_x_label/train**  and another for **autoencoder_y_label/train** and please change the path of the the **root directory** as the directory containing these folders. The dataset should be structured as follows:

```
dataset/
├── autoencoder_x_label/train/
│   └── image1.jpg
│   └── image2.jpg
├── autoencoder_y_label/train/
│   └── image1.jpg
│   └── image2.jpg
```

You will need to implement a dataset class that loads these images properly. The `PerturbedDataset` class handles this by checking for directory existence and ensuring that there is a matching number of files in both directories.

## Training and Evaluation
To train the model:
1. Load your dataset using the `PerturbedDataset` class.
2. Define your training loop including loss calculation and optimization steps.
3. Use metrics such as F1 score or accuracy for evaluation.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

This README provides a comprehensive overview of the Autoencoder model's functionality and usage. For further details on implementation specifics or additional features, refer to the source code within this notebook.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/11448647/bdaefd89-e331-46f3-963e-f8c7eaae2d05/autoencoder.ipynb
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/11448647/21d2fdd1-ca61-47db-8951-04386f520306/HolloPatch.ipynb