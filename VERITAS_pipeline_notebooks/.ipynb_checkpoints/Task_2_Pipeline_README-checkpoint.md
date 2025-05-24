
# Task 2 Pipeline

## Overview

The Task 2 pipeline is designed to detect artifacts in fake images and generate detailed descriptions for each detected artifact. The pipeline consists of two Jupyter notebooks that need to be executed sequentially to produce the final results.

---

## Prerequisites

### Folder Structure

Ensure that the following folder structure is created in the same directory as the Jupyter notebooks:

```
|-- Task_2_Pipeline/
    |-- Artifact_detection.ipynb
    |-- Generate_descriptions.ipynb
    |-- Model_weights/
        |-- <Place Grad-CAM model weights and DRCT weights here>
    |-- Artifact_descriptions/
        |-- Artifact_description_Tuple.csv
    |-- input_imgs/
        |-- <Place fake images to be processed here>
```

### Requirements

1. Grad-CAM model weights and DRCT weights should be placed in the `Model_weights/` folder.
2. The artifact descriptions CSV file (`Artifact_description_Tuple.csv`) should be placed in the `Artifact_descriptions/` folder.
3. Place all fake images to be tested in the pipeline inside the `input_imgs/` folder.

---

## Steps to Run the Pipeline

### Step 1: Run `Artifact_detection.ipynb`

1. Open the `Artifact_detection.ipynb` notebook in your Jupyter editor.
2. Run all cells (select **Run All** from the **Cell** menu).
3. This notebook will:
   - Generate super-resolution images.
   - Detect artifacts for each image.
   - Save the results in a JSON file named `Artifact_detected.json`.

### Step 2: Run `Generate_descriptions.ipynb`

1. Once `Artifact_detected.json` is generated, open the `Generate_descriptions.ipynb` notebook.
2. Run all cells (select **Run All** from the **Cell** menu).
3. This notebook will:
   - Process the artifacts detected in each image.
   - Generate detailed descriptions for each artifact.
   - Save the final results in a JSON file named `artifacts_with_description_submission.json`.

---

## Outputs

1. **`Artifact_detected.json`**:
   - Contains detected artifacts for each image.
2. **`artifacts_with_description_submission.json`**:
   - Contains detected artifacts along with their descriptions for each image.

---

## Notes

- The pipeline is designed to work on **fake images only**. Ensure that only fake images are placed in the `input_imgs/` folder.
- If any issues arise during execution, ensure all dependencies are installed and the required files are in their respective folders.
