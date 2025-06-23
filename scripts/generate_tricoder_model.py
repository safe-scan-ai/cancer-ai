import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleSkinLesionModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input shape: [batch_size, 3, 224, 224]
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 112x112
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 56x56
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 28x28
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def export_to_onnx(output_path='sample_tricorder_model.onnx'):
    # Create model and set to evaluation mode
    model = SimpleSkinLesionModel(num_classes=11)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export the model
    torch.onnx.export(
        model,                     # model being run
        dummy_input,               # model input
        output_path,               # output file
        export_params=True,        # store the trained parameter weights
        opset_version=11,          # ONNX version
        do_constant_folding=True,  # optimize the model
        input_names=['input'],     # model's input names
        output_names=['output'],   # model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    export_to_onnx()
