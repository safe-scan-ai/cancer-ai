import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image
import argparse

# Tricorder-3 class mapping (11 classes)
CLASS_NAMES = [
    "Actinic keratosis/intraepidermal carcinoma (AKIEC)",
    "Basal cell carcinoma (BCC)",
    "Other benign proliferations (BEN_OTH)",
    "Benign keratinocytic lesion (BKL)",
    "Dermatofibroma (DF)",
    "Inflammatory and infectious (INF)",
    "Other malignant proliferations (MAL_OTH)",
    "Melanoma (MEL)",
    "Melanocytic Nevus (NV)",
    "Squamous cell carcinoma/keratoacanthoma (SCCKA)",
    "Vascular lesions and hemorrhage (VASC)"
]

class ONNXInference:
    def __init__(self, model_path):
        """Initialize ONNX model session and detect input size."""
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        
        # Detect model's expected input size from first input
        self.input_size = self._get_input_size()
        print(f"Model input size: {self.input_size}")
    
    def _get_input_size(self):
        """Extract expected image input size from ONNX model."""
        inputs = self.session.get_inputs()
        if inputs:
            shape = inputs[0].shape
            # Shape is typically [batch_size, channels, height, width]
            if len(shape) >= 4:
                h = shape[2] if isinstance(shape[2], int) else 512
                w = shape[3] if isinstance(shape[3], int) else 512
                return (h, w)
        return (512, 512)  # Default fallback
    
    def preprocess_image(self, image_path):
        """Load and preprocess image to [0, 1] range."""
        img = Image.open(image_path).convert('RGB')
        
        # Resize to model's expected size
        h, w = self.input_size
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to BCHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path, age, gender, location):
        """Run inference on a single image with demographic data."""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Convert demographics to proper format
        # Gender: 'm' -> 1.0, 'f' -> 0.0
        gender_encoded = 1.0 if gender.lower() == 'm' else 0.0
        
        # Prepare demographic data as [age, gender_encoded, location]
        demo_tensor = np.array([[float(age), gender_encoded, float(location)]], dtype=np.float32)
        
        # Run inference
        inputs = {self.input_names[0]: image_tensor, self.input_names[1]: demo_tensor}
        outputs = self.session.run(None, inputs)
        
        # Model already outputs probabilities (softmax applied in forward pass)
        probs = outputs[0].flatten()
        
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
    parser.add_argument('--age', type=int, required=True,
                        help='Patient age in years (e.g., 42)')
    parser.add_argument('--gender', type=str, required=True, choices=['m', 'f'],
                        help='Patient gender: m (male) or f (female)')
    parser.add_argument('--location', type=int, required=True, choices=range(1, 8),
                        help='Body location: 1=Arm, 2=Feet, 3=Genitalia, 4=Hand, 5=Head, 6=Leg, 7=Torso')
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
        predictions = model.predict(args.image, args.age, args.gender, args.location)
        
        location_names = {1: "Arm", 2: "Feet", 3: "Genitalia", 4: "Hand", 5: "Head", 6: "Leg", 7: "Torso"}
        print(f"Demographics: Age={args.age}, Gender={args.gender.upper()}, Location={location_names[args.location]}")
        print("\nTop 3 Predictions:")
        print("-" * 40)
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {prob*100:.2f}%")
            
    except RuntimeError as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()
