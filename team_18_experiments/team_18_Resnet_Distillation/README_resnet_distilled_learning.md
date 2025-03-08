# ResNet Distillation Learning

## Overview
This implementation is a *ResNet-based model* for image classification using *knowledge distillation*. The framework consists of a teacher model that guides the training of a student model, leveraging adversarial training techniques to enhance performance.

## Requirements
Install dependencies:
bash
pip install -r requirements.txt


## Project Structure
- ResNet_Distillation_learning.ipynb: Main Jupyter notebook with implementation
- requirements.txt: Project dependencies
- Data/: Directory for training and testing datasets

## Installation
1. Clone the repository:
   bash
   git clone <repository-url>
   cd <repository-directory>
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Prepare dataset:
   - Training images: /path/to/dataset/train/
     - Organize images by class (e.g., class1/, class2/)
   - Test images: /path/to/dataset/test/
     - Similar structure to training directory

## Usage

### Data Preparation
Unzip dataset:
python
unzip_local_file(zip_file_path, dest_folder)


### Training Models
1. Train Teacher Model:
   python
   train_model_teacher(teacher, train_dataloader, val_dataloader, epochs=25, lr=0.0001)
   

2. Train Student Model:
   python
   train_model_student_jd(student, teacher, train_dataloader, test_dataloader, aug_fn, epochs=25, lr=0.0001)
   

3. Evaluate Models:
   python
   evaluate(model, test_dataloader)  # Generates accuracy and classification report
   

## Adversarial Training Techniques
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)

## Model Architecture
### Teacher Model
- Custom ResNet encoder
- Two-class output
- Modified convolutional layers

### Student Model
- ResNet-based architecture
- Trained via knowledge distillation to mimic teacher's outputs

## Key Functions
- unzip_local_file(): Extract zip archives
- train_model_teacher(): Train teacher model with early stopping
- train_model_student_jd(): Implement knowledge distillation
- evaluate(): Model performance assessment

## Performance Recommendations
- Use CUDA-enabled GPU
- Adjust hyperparameters based on:
  - Dataset characteristics
  - Hardware capabilities