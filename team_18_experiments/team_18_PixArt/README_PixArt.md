# PixArt Sigma Image Generation

This project demonstrates how to use the `PixArtSigmaPipeline` from the Hugging Face `diffusers` library to generate an image from a text prompt using a pre-trained model. The model is loaded onto the GPU if available, or CPU otherwise, and generates images in the form of `PIL` images.

## Requirements

- Python 3.7+
- PyTorch (with CUDA support for GPU acceleration, if applicable)
- Hugging Face Diffusers library
- Other dependencies: `torch`, `diffusers`

### Installation

1. Install the required dependencies:

```bash
pip install torch diffusers safetensors
