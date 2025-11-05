import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleSkinLesionModel(nn.Module):
    def __init__(self, num_classes=11, num_demographics=3):
        super().__init__()
        # Image feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x256

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Demographic data processor
        self.demographics_processor = nn.Sequential(
            nn.Linear(num_demographics, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        # Combined classifier
        self.classifier = nn.Linear(64 + 16, num_classes, bias=False)

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
    model = SimpleSkinLesionModel(num_classes=11, num_demographics=3)
    model.eval()

    # Create dummy inputs
    dummy_image = torch.randn(1, 3, 512, 512)
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
    print(f"- Classes: 11 (vs 10 for Tricorder-2)")

if __name__ == "__main__":
    export_optimized_model()
