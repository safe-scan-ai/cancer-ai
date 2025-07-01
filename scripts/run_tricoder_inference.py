import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Class mapping for tricorder competition
CLASS_NAMES = {
    0: "Actinic Keratosis (AK)",
    1: "Basal Cell Carcinoma (BCC)",
    2: "Seborrheic Keratosis (SK)",
    3: "Squamous Cell Carcinoma (SCC)",
    4: "Vascular Lesion (VASC)",
    5: "Dermatofibroma (DF)",
    6: "Benign Nevus (NEVUS)",
    7: "Other Non-Neoplastic (ONN)",
    8: "Melanoma (MEL)",
    9: "Other Neoplastic (ON)",
    10: "Benign (BENIGN)"
}

class ONNXInference:
    def __init__(self, model_path):
        """Initialize ONNX model session."""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        return self.transform(img).unsqueeze(0).numpy()
    
    def predict(self, image_path):
        """Run inference on a single image."""
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Get predictions (softmax to get probabilities)
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
        probs = probs.flatten()
        
        # Get top 3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
        
        return top3

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with ONNX model')
    parser.add_argument('--model', type=str, default='sample_models/sample_tricorder_model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    # Initialize and run inference
    print(f"\nLoading model: {args.model}")
    print(f"Processing image: {args.image}\n")
    
    try:
        model = ONNXInference(args.model)
        predictions = model.predict(args.image)
        
        print("Top 3 Predictions:")
        print("-" * 40)
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()
