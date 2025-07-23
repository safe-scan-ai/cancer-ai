#!/bin/bash

# Exit on error
set -e

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
MODEL_DIR="${ROOT_DIR}/models"
OUTPUT_DIR="${ROOT_DIR}/output"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to print section headers
section() {
    echo -e "\n=== $1 ==="
}

# Step 1: Verify example dataset
section "Checking Example Dataset"
echo "Using example dataset from ${SCRIPT_DIR}/example_dataset"
echo "Images: $(ls -1 "$SCRIPT_DIR/example_dataset/"*.jpg 2>/dev/null | wc -l)"
echo "Labels: $(tail -n +2 "$SCRIPT_DIR/example_dataset/label.csv" 2>/dev/null | wc -l) entries"

# Step 2: Generate model
section "Generating Model"
python3 "${SCRIPT_DIR}/generate_tricorder_model.py"

# Step 3: Verify model files exist
section "Verifying Model Files"
if [ -f "${ROOT_DIR}/sample_tricorder_model.pt" ] && [ -f "${ROOT_DIR}/sample_tricorder_model.onnx" ]; then
    echo " Model files generated successfully"
    ls -lh "${ROOT_DIR}/sample_tricorder_model.pt" "${ROOT_DIR}/sample_tricorder_model.onnx"
else
    echo " Error: Model files not found!"
    exit 1
fi

# Step 4: Run inference on a sample image
section "Running Inference"
SAMPLE_IMAGE=$(find "${SCRIPT_DIR}/example_dataset" -type f -name "*.jpg" | head -n 1)
if [ -z "$SAMPLE_IMAGE" ]; then
    echo " Error: No sample images found in ${SCRIPT_DIR}/example_dataset"
    exit 1
fi

echo "Using sample image: $SAMPLE_IMAGE"
python3 "${SCRIPT_DIR}/run_tricorder_inference.py" \
    --model "${ROOT_DIR}/sample_tricorder_model.onnx" \
    --image "$SAMPLE_IMAGE" \
    --age 42 \
    --gender f \
    --location 7

# Print completion message
section "Pipeline Completed Successfully"
echo " All steps completed successfully!"
echo -e "\n Outputs:"
echo "- Example dataset: ${SCRIPT_DIR}/example_dataset"
echo "- Model files: ${ROOT_DIR}/sample_tricorder_model.{pt,onnx}"
