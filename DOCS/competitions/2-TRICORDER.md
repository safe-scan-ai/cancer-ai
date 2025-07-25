# 🏆 Competition: Skin Lesion Classification Based on Images

## 🎯 Competition Goal

The goal of the competition is to build a lightweight and effective ML model that classifies skin lesions into one of 10 predefined disease classes based on lesion images and demographic data.

## 📥 Input and Output Data

### Input

#### 1. Skin Lesion Image

- **Format**: JPEG or PNG
- **Channels**: RGB (3 channels), no alpha channel
- **Minimum side length**: ≥ 512 px
- **Pixel values**: range [0, 512], dtype=np.uint8

#### 2. Patient Demographic Data

- **Age**: integer in years (e.g., 42)
- **Gender**: "m" (male) / "f" (female)
- **Body location**: integer according to the table below

> **Note**: The model must utilize both image and demographic data.

### Output

- **List of 10 class probabilities**: List[float]
- **Probabilities must sum to 1.0** (softmax)
- **Value range**: [0.0, 1.0]

## 🧬 Class List (order in model output)

| No. | Class | Clinical Type | Symbol |
|:---|:---|:---|:---|
| 1 | Actinic keratosis (AK) | Benign | AK |
| 2 | Basal cell carcinoma (BCC) | Malignant | BCC |
| 3 | Seborrheic keratosis (SK) | Medium risk | SK |
| 4 | Squamous cell carcinoma (SCC) | Malignant | SCC |
| 5 | Vascular lesion | Medium risk | VASC |
| 6 | Dermatofibroma | Benign | DF |
| 7 | Benign nevus | Benign | NV |
| 8 | Other non-neoplastic | Benign | NON |
| 9 | Melanoma | Malignant | MEL |
| 10 | Other neoplastic / Benign | Benign | ON |

## ⚖️ Class Weights

| Class Type | Classes (No.) | Color | Weight |
|:---|:---|:---:|:---|
| Malignant | 2, 4, 9 | 🔴 | 3× (BCC, SCC, MEL) |
| Medium risk | 3, 5 | 🟠 | 2× (SK, VASC) |
| Benign | 1, 6, 7, 8, 10 | 🟢 | 1× (AK, DF, NV, NON, ON) |

## 📍 Body Location List

| No. | Location |
|:---|:---|
| 1 | Arm |
| 2 | Feet |
| 3 | Genitalia |
| 4 | Hand |
| 5 | Head |
| 6 | Leg |
| 7 | Torso |

## 🧮 Evaluation Criteria (100 pts)

| Category | Weight | Max pts | Notes |
| :--- | :--- | :--- | :--- |
| Prediction Quality | 90% | 90 pts | Weighted average: 40% Accuracy, 60% Weighted-F1. This score is then penalized by the misclassification cost. |
| Efficiency | 10% | 10 pts | Based on model size. See formula below. |

## 💎 Misclassification Cost

To prioritize patient safety, a cost matrix is used to penalize misclassifications based on their clinical severity. The cost is higher for more dangerous errors (e.g., classifying a high-risk lesion as benign).

| True Risk | Predicted Benign | Predicted Medium Risk | Predicted High Risk |
| :--- | :--- | :--- | :--- |
| **Benign** | 0 | 1 | 5 |
| **Medium Risk** | 10 | 0 | 2 |
| **High Risk** | 50 | 20 | 0 |

## 📊 Score Calculation

### F1-score for class types

```text
F1_malignant = (F1_2 + F1_4 + F1_9) / 3  
F1_medium    = (F1_3 + F1_5) / 2  
F1_benign    = (F1_1 + F1_6 + F1_7 + F1_8 + F1_10) / 5
```

### Weighted-F1

```text
Weighted-F1 = (3 × F1_malignant + 2 × F1_medium + 1 × F1_benign) / 6
```

### Prediction Score

```text
Prediction Score = (0.4 × Accuracy) + (0.6 × Weighted-F1)
```

### Efficiency Score

```text
Efficiency Score = 1 - (ModelSize - MinSize) / (MaxSize - MinSize)
```

### Final Score

The final score is calculated by taking a weighted average of the prediction and efficiency scores, and then applying the `misclassification_cost` as a multiplicative penalty.

```text
Base Score = (0.9 * Prediction Score) + (0.1 * Efficiency Score)
Normalized Cost = TotalMisclassificationCost / MaxPossibleCost
Final Score = Base Score * (1 - Normalized Cost)
```

## 💡 Additional Notes

- Models may return high probabilities for multiple classes – this will not be penalized as long as softmax is correct.
- Calibration is not required but may improve prediction usefulness.
- Models with size < 50 MB receive maximum points for size in efficiency scoring.

## 🔧 Example Implementation

Example scripts and pipeline available in: `DOCS/competitions/tricorder_samples/`

### Running the example

```bash
cd DOCS/competitions/tricorder_samples
./run_pipeline.sh
```

### Example structure

- `generate_tricorder_model.py` - 10-class model generation
- `run_tricorder_inference.py` - Inference script with demographic data
- `example_dataset/` - Sample dataset with images and labels
- `README_EXAMPLE_TRICORDER.md` - Detailed documentation

## 📋 Submission Requirements

- Model must accept both image and demographic inputs
- Output exactly 10 probabilities that sum to 1.0
- Model size should be optimized (< 150 MB, ideally < 50 MB)
- Include inference script compatible with the evaluation framework
