# Tricorder Model Pipeline

This directory contains all scripts needed for the Tricorder skin lesion classification model, including data loading, model generation, verification, and inference.

## Prerequisites

- Python 3.8+
- Required Python packages (install with `pip install -r requirements.txt` from project root)
- Bash shell (for running the pipeline script)

## Directory Structure

```
DOCS/competitions/tricorder_samples/
├── run_pipeline.sh              # Main pipeline script
├── example_dataset/             # Sample dataset with images and labels
│   ├── *.jpg                   # Sample skin lesion images
│   └── label.csv               # Labels with demographics
├── generate_tricorder_model.py # Model generation (10-class with demographics)
├── run_tricorder_inference.py  # Inference script with new demographic format
```

## Usage

1. Make the script executable (if not already):
   ```bash
   chmod +x run_pipeline.sh
   ```

2. Run the pipeline:
   ```bash
   ./run_pipeline.sh
   ```

The pipeline will execute the following steps:

1. **Load Test Data**: Copies sample data from `example_dataset/` to the `data/` directory
2. **Generate Model**: Creates the Tricorder model files
3. **Verify Model Files**: Checks that model files were created successfully
4. **Run Inference**: Tests the model with a sample image and demographic data (also validates model functionality)

## Outputs

- Model files will be saved in the root directory:
  - `sample_tricorder_model.pt`: PyTorch model
  - `sample_tricorder_model.onnx`: ONNX model
- Sample data will be saved in the `data/` directory
- Output and logs will be displayed in the console

## Running Custom Inference

To run inference on a custom image with specific demographic data:

```bash
python3 ./run_tricorder_inference.py \
    --model ../../../sample_tricorder_model.onnx \
    --image /path/to/your/image.jpg \
    --age <age> \
    --gender <m|f> \
    --location <1-7>
```

Example:
```bash
python3 ./run_tricorder_inference.py \
    --model ../../../sample_tricorder_model.onnx \
    --image ../../../data/images/sample.jpg \
    --age 42 \
    --gender f \
    --location 7
```

## Demographic Parameters

- **Age**: Integer years (e.g., 42)
- **Gender**: "m" (male) or "f" (female)
- **Location**: Integer 1-7 mapping to:
  - 1: Arm
  - 2: Feet  
  - 3: Genitalia
  - 4: Hand
  - 5: Head
  - 6: Leg
  - 7: Torso

## Model Specifications

- **Input**: 512x512 RGB image with pixel values in [0,512] range + demographics
- **Output**: 10 class probabilities (sum to 1.0)
- **Classes**: AK, BCC, SK, SCC, VASC, DF, NV, NON, MEL, ON

## Notes

- The script will create necessary directories if they don't exist
- All paths are relative to the project root (3 levels up from this directory)
- Model files are saved in the project root directory
- Sample dataset is included in the `example_dataset/` subdirectory
- For production use, consider adding proper error handling and logging
