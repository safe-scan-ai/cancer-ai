import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleSkinLesionModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # Smaller model with depthwise separable convolutions
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        self.classifier = nn.Linear(64, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def export_optimized_model(output_path='sample_tricorder_model.pt'):
    # Create model and set to evaluation mode
    model = SimpleSkinLesionModel(num_classes=11)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX with optimization
    onnx_path = output_path.replace('.pt', '.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Save the model in FP16
    model = model.half()
    dummy_input = dummy_input.half()
    scripted_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(scripted_model, output_path)
    
    # Print model size information
    import os
    pt_size = os.path.getsize(output_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Optimized model sizes:")
    print(f"- PyTorch (FP16): {pt_size:.2f} MB")
    print(f"- ONNX: {onnx_size:.2f} MB")

if __name__ == "__main__":
    export_optimized_model()
