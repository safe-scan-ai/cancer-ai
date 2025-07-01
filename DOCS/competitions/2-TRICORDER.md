# Description of Tricorder Skin Lesion Classification Competition

## Overview
This competition challenges participants to develop a **lightweight and accurate** machine learning model for **multi-class skin lesion classification**. The model will analyze skin lesion images along with patient demographic data to classify them into one of 11 diagnostic categories.

### Objective
The primary goal is to create a model that can assist in the early detection and classification of various skin conditions, from benign lesions to malignant melanomas. The model should provide reliable predictions while being efficient enough to run on mobile devices, making it accessible for preliminary screening.

## Evaluation Criteria
Models will be evaluated based on a combination of **prediction quality** and **efficiency metrics**.

### Performance Metrics (90% weight)

| **Metric**       | **Description**                                                                 | **Weight** |
|------------------|---------------------------------------------------------------------------------|------------|
| **Accuracy**     | Overall correctness of predictions across all classes                           | 45%        |
| **Weighted F1**  | F1 score weighted by risk categories (higher weight for more serious conditions) | 45%        |
| **Efficiency**   | Model size and inference speed metrics (see below)                               | 10%        |


### Efficiency Metrics (10% weight)

| **Metric**            | **Description**                                                                 | **Weight** |
|-----------------------|---------------------------------------------------------------------------------|------------|
| **Model Size**       | Smaller models score higher (target < 50MB for full score)                      | 5%         |
| **Inference Speed**  | Faster inference times score higher (relative to other submissions)           | 5%         |

### Risk-Based Weighting

Predictions are weighted by clinical significance:

- **High Risk (3× weight)**: Basal Cell Carcinoma, Squamous Cell Carcinoma, Melanoma
- **Medium Risk (2× weight)**: Seborrheic Keratosis, Vascular Lesion
- **Low Risk (1× weight)**: Actinic Keratosis, Dermatofibroma, Benign Nevus, Other Non-Neoplastic, Other Neoplastic, Benign

## Output Format
Models should output a probability distribution over the 10 classes that sums to 1.0, with each value in the range [0.0, 1.0].

## Class Definitions

| ID | Class Name | Short Name | Risk Category | Weight |
|----|--------------------------------|------------|----------------|--------|
| 1  | Actinic Keratosis (AK)         | AK         | Low Risk       | 1.0    |
| 2  | Basal Cell Carcinoma (BCC)     | BCC        | High Risk      | 3.0    |
| 3  | Seborrheic Keratosis (SK)      | SK         | Medium Risk    | 2.0    |
| 4  | Squamous Cell Carcinoma (SCC)  | SCC        | High Risk      | 3.0    |
| 5  | Vascular Lesion               | VASC       | Medium Risk    | 2.0    |
| 6  | Dermatofibroma                | DF         | Low Risk       | 1.0    |
| 7  | Benign Nevus                  | NEVUS      | Low Risk       | 1.0    |
| 8  | Other Non-Neoplastic          | ONN        | Low Risk       | 1.0    |
| 9  | Melanoma                      | MEL        | High Risk      | 3.0    |
| 10 | Other Neoplastic              | ON         | Low Risk       | 1.0    |
| 11 | Benign                       | BENIGN     | Low Risk       | 0.5    |

## Technical Requirements
- Input: RGB image (min 512px on shortest side) + patient age and gender
- Output: 10-class probability distribution
- Model size: Target < 150MB (50MB for full efficiency points)
- Framework: Any (ONNX, TensorFlow, PyTorch, etc.)

## Intended Use
The winning model will be integrated into a mobile application to assist healthcare professionals in preliminary skin lesion assessment, particularly in resource-constrained settings.
