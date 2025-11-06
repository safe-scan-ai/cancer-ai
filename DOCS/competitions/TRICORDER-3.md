# 🏆 Competition: Skin Lesion Classification Based on Images

## 🎯 Competition Goal

The goal of the competition is to build a lightweight and effective ML model that classifies skin lesions into one of 11 predefined disease classes based on lesion images and demographic data.

(Note that it's the same as [TRICORDER-2](TRICORDER-2.md), but with 11 classes aligned from ISIC-2025 competition, instead of 10.)

## 📥 Input and Output Data

### Input

#### 1. Skin Lesion Image

- **Format**: JPEG or PNG
- **Channels**: RGB (3 channels), no alpha channel
- **Minimum side length**: ≥ 512 px
- **Preprocessing**: Resize to model's expected size (typically 512×512), normalize to [0, 1] range

#### 2. Patient Demographic Data

- **Age**: integer in years (e.g., 42)
- **Gender**: "m" (male) / "f" (female)
- **Body location**: integer according to the table below

> **Note**: The model must utilize both image and demographic data.

### Output

- **List of 11 class probabilities**: List[float]
- **Probabilities must sum to 1.0** (softmax)
- **Value range**: [0.0, 1.0]

## 🧬 Class List (order in model output)

| No. | Class | Clinical Type | Symbol |
|-----|-------|---------------|--------|
| 1 | Actinic keratosis/intraepidermal carcinoma | Medium risk | AKIEC |
| 2 | Basal cell carcinoma | Malignant | BCC |
| 3 | Other benign proliferations including collisions | Benign | BEN_OTH |
| 4 | Benign keratinocytic lesion | Medium risk | BKL |
| 5 | Dermatofibroma | Benign | DF |
| 6 | Inflammatory and infectious | Benign | INF |
| 7 | Other malignant proliferations including collisions | Malignant | MAL_OTH |
| 8 | Melanoma | Malignant | MEL |
| 9 | Melanocytic Nevus, any type | Benign | NV |
| 10 | Squamous cell carcinoma/keratoacanthoma | Malignant | SCCKA |
| 11 | Vascular lesions and hemorrhage | Medium risk | VASC |

[🧬 Full detailed disease classes mapping](https://github.com/safe-scan-ai/cancer-ai/blob/main/DOCS/competitions/TRICORDER-3-DISEASE-MAPPING.md)

## ⚖️ Class Weights

| Class Type | Classes (No.) | Color | Weight |
|------------|---------------|-------|--------|
| Malignant | 2, 7, 8, 10 | 🔴 | 3× (BCC, MAL_OTH, MEL, SCCKA) |
| Medium risk | 1, 4, 11 | 🟠 | 2× (AKIEC, BKL, VASC) |
| Benign | 3, 5, 6, 9 | 🟢 | 1× (BEN_OTH, DF, INF, NV) |

## 📍 Body Location List

| No. | Location |
|-----|----------|
| 1 | Arm |
| 2 | Feet |
| 3 | Genitalia |
| 4 | Hand |
| 5 | Head |
| 6 | Leg |
| 7 | Torso |

## 🧮 Evaluation Criteria (100 pts)

| Category | Weight | Max pts | Notes |
|----------|--------|---------|-------|
| Prediction Quality | 90% | 90 pts | Weighted average: 50% Accuracy, 50% Weighted-F1 |
| Efficiency | 10% | 10 pts | Model size (50%) + inference speed (50%) |

## 📊 Score Calculation

### F1-score for class types

```text
F1_malignant = (F1_2 + F1_7 + F1_8 + F1_10) / 4  
F1_medium    = (F1_1 + F1_4 + F1_11) / 3  
F1_benign    = (F1_3 + F1_5 + F1_6 + F1_9) / 4
```

### Weighted-F1

```text
Weighted-F1 = (3 × F1_malignant + 2 × F1_medium + 1 × F1_benign) / 6
```

### Accuracy

Standard top-1 classification accuracy (percentage of correct classifications)

### Prediction Score (90%)

```text
Prediction Score = 0.5 × Accuracy + 0.5 × Weighted-F1
```

### Efficiency Score

```text
Efficiency Score = 0.5 × Size Score + 0.5 × Speed Score
```

**Size Score:**

```text
Size Score = 1.0  if model_size ≤ 50 MB
Size Score = (150 - model_size) / 100  if 50 MB < model_size ≤ 150 MB
Size Score = 0.0  if model_size > 150 MB
```

**Speed Score:**

```text
Speed Score = (slowest_time - model_time) / (slowest_time - fastest_time)
```

- Fastest model gets score of 1.0
- Slowest model gets score of 0.0
- Models with equal speed all get score of 1.0

**Where:**

- **model_size** – model size in MB
- **model_time** – average inference time per image (in ms)
- **fastest_time** – fastest inference time in competition
- **slowest_time** – slowest inference time in competition
- **Efficiency Score ∈ [0.0, 1.0]**

> **Note**: Inference time will be measured on uniform CPU hardware (no GPU) and averaged across all test images.

### Final Score

```text
Final Score = 0.9 × Prediction Score + 0.1 × Efficiency Score
```

## 💡 Additional Notes

- Models may return high probabilities for multiple classes – this will not be penalized as long as softmax is correct.
- Calibration is not required but may improve prediction usefulness.
- Models with size < 50 MB receive maximum points for size in efficiency scoring.

## 🔧 Example Implementation

Example scripts and pipeline available in: `DOCS/competitions/tricorder_samples/`

### Model Architecture Reference

The model architecture and constants are defined in:
- **Architecture**: `cancer_ai/validator/competition_handlers/tricorder_common.py` (lines 51-57)
  - `TRICORDER_3_IMAGE_SIZE = (512, 512)`
  - `TRICORDER_3_NUM_CLASSES = 11`
  - `TRICORDER_3_FEATURE_CHANNELS = [16, 32, 64]`
  - `TRICORDER_3_DEMOGRAPHICS_FEATURES = 16`

### Running the example

```bash
cd DOCS/competitions/tricorder_samples
python generate_tricorder_3_model.py  # Generate sample model
python run_tricorder_inference.py --model sample_tricorder_3_model.onnx --image <path> --age 42 --gender m --location 5
```

### Example structure

- `generate_tricorder_3_model.py` - 11-class model generation (uses constants from tricorder_common.py)
- `run_tricorder_inference.py` - Inference script with demographic data
- `example_dataset/` - Sample dataset with images and labels
- `README_EXAMPLE_TRICORDER.md` - Detailed documentation

## 📋 Submission Requirements

- Model must accept both image and demographic inputs
- Output exactly 11 probabilities that sum to 1.0
- Model size should be optimized (< 150 MB, ideally < 50 MB)
- Include inference script compatible with the evaluation framework
