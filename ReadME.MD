# Decoding Real and AI-Generated: An Explainable Approach to Image Detection

## Overview
The proliferation of AI-generated content has necessitated the development of robust methodologies for distinguishing real images from synthetic ones. This project introduces a two-stage framework for AI-generated image detection. The first stage employs a high-accuracy classification model to differentiate between real and synthetic images. The second stage integrates an interpretability framework that identifies and highlights visual artifacts such as texture inconsistencies, lighting anomalies, and structural deviations, providing transparency in decision-making.

## Features
- **Robust Image Classification**: Implements state-of-the-art deep learning models for accurate differentiation between real and AI-generated images.
- **Artifact Detection and Explainability**: Utilizes advanced interpretability techniques to highlight and analyze distinguishing features.
- **Adversarial Robustness**: Enhances model resilience against adversarial perturbations through defensive training strategies.
- **Super-Resolution for Detailed Analysis**: Applies super-resolution techniques to improve artifact detection in low-resolution images.

## Methodology
### 1. Classification Framework
- Utilizes an ensemble of EfficientNet, ResNet-18, and ViT-Tiny for robust classification.
- Trained on an augmented CIFAKE dataset, incorporating adversarially modified images.
- Implements a weighted ensemble approach to optimize predictive performance.

### 2. Artifact Detection Mechanism
- Enhances image resolution using DRCT Super-Resolution models.
- Applies GradCAM-based heatmaps to visualize decision-critical regions.
- Utilizes CLIP-based scoring to quantify and rank unnatural features.
- Leverages MOLMO to generate textual descriptions of detected artifacts, improving interpretability.

## Pipelines
### 1. Task 1 Pipeline
![Pipeline goes here](relative%20path/to/img.jpg?raw=true "Title")
### 2. Task 2 pipeline
![Task 2 pipeline](pipeline_structures/Task_2_pipeline_structure.png?raw=true "Task 2 Pipeline")

## Dataset Composition
The dataset builds upon the CIFAKE dataset, integrating adversarial perturbations from:
- **Fast Gradient Sign Method (FGSM)**: Introduces minimal perturbations to deceive classifiers.
- **Projected Gradient Descent (PGD)**: Iteratively refines perturbations for enhanced evasion.
- **AutoAttack**: Employs an ensemble of adversarial attack strategies.
- **AuraSR**: Includes images generated via GAN-based techniques to improve dataset diversity.

## Model Performance
| Model               | Accuracy (%) |
|---------------------|-------------|
| EfficientNet        | 87.7        |
| ResNet-18          | 83.5        |
| ViT-Tiny           | 83.1        |
| **Ensemble Model** | **93.1**    |

## Future Enhancements
- Aggregate GradCAM intensity maps from multiple models to improve artifact localization.
- Implement hallucination correction techniques to refine MOLMO-generated textual descriptions.
- Explore advanced resizing techniques to mitigate interpolation artifacts.
- Strengthen resilience to adversarial perturbations using auto-LiRPA for certified robustness.

## Key Challenges
- Ensuring robustness against adversarially manipulated images in real-world scenarios.
- Addressing variations in synthetic image quality across different generative models.
- Overcoming the limitations posed by low-resolution 32x32 image inputs.

