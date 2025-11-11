import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directories to path to import from cancer_ai
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from cancer_ai.validator.competition_handlers.tricorder_common import (
    TRICORDER_3_NUM_CLASSES,
    TRICORDER_3_NUM_DEMOGRAPHICS,
    TRICORDER_3_IMAGE_SIZE,
    TRICORDER_3_FEATURE_CHANNELS,
    TRICORDER_3_DEMOGRAPHICS_FEATURES,
    TRICORDER_3_COMBINED_FEATURES,
)

class SimpleSkinLesionModel(nn.Module):
    def __init__(self, num_classes=TRICORDER_3_NUM_CLASSES, num_demographics=TRICORDER_3_NUM_DEMOGRAPHICS):
        super().__init__()
        # Image feature extractor with channels from constants
        ch = TRICORDER_3_FEATURE_CHANNELS  # [16, 32, 64]
        self.features = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x256

            nn.Conv2d(ch[0], ch[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128

            nn.Conv2d(ch[1], ch[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Demographic data processor
        self.demographics_processor = nn.Sequential(
            nn.Linear(num_demographics, TRICORDER_3_DEMOGRAPHICS_FEATURES),
            nn.ReLU(),
            nn.BatchNorm1d(TRICORDER_3_DEMOGRAPHICS_FEATURES)
        )

        # Combined classifier
        self.classifier = nn.Linear(TRICORDER_3_COMBINED_FEATURES, num_classes, bias=False)

    def forward(self, image, demographics):
        image_features = self.features(image)
        image_features = image_features.view(image_features.size(0), -1)

        demographics_features = self.demographics_processor(demographics)

        combined_features = torch.cat((image_features, demographics_features), dim=1)

        logits = self.classifier(combined_features)
        # Apply softmax to get probabilities that sum to 1.0
        probabilities = F.softmax(logits, dim=1)
        return probabilities

def export_optimized_model(output_path='sample_tricorder_3_model.pt'):
    # Create model and set to evaluation mode
    model = SimpleSkinLesionModel()
    model.eval()

    # Create dummy inputs using constants
    h, w = TRICORDER_3_IMAGE_SIZE
    dummy_image = torch.randn(1, 3, h, w)
    dummy_demo = torch.tensor([[42.0, 1.0, 5.0]])  # age, sex, location

    # Export to ONNX with optimization
    onnx_path = output_path.replace('.pt', '.onnx')
    torch.onnx.export(
        model,
        (dummy_image, dummy_demo),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['image', 'demographics'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'demographics': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Save the model in FP16
    model = model.half()
    dummy_image = dummy_image.half()
    dummy_demo = dummy_demo.half()
    scripted_model = torch.jit.trace(model, (dummy_image, dummy_demo))
    torch.jit.save(scripted_model, output_path)
    
    # Print model size information
    import os
    pt_size = os.path.getsize(output_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Tricorder-3 optimized model sizes:")
    print(f"- PyTorch (FP16): {pt_size:.2f} MB")
    print(f"- ONNX: {onnx_size:.2f} MB")
    print(f"- Input size: {TRICORDER_3_IMAGE_SIZE}")
    print(f"- Classes: {TRICORDER_3_NUM_CLASSES}")
    print(f"- Demographics features: {TRICORDER_3_DEMOGRAPHICS_FEATURES}")
    print(f"- Feature channels: {TRICORDER_3_FEATURE_CHANNELS}")

if __name__ == "__main__":
    export_optimized_model()
