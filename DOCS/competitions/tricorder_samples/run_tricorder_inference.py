import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import argparse

# Class mapping for tricorder competition
CLASS_NAMES = [
    "Actinic keratosis (AK)",
    "Basal cell carcinoma (BCC)", 
    "Seborrheic keratosis (SK)",
    "Squamous cell carcinoma (SCC)",
    "Vascular lesion (VASC)",
    "Dermatofibroma (DF)",
    "Benign nevus (NV)",
    "Other non-neoplastic (NON)",
    "Melanoma (MEL)",
    "Other neoplastic (ON)"
]

class ONNXInference:
    def __init__(self, model_path):
        """Initialize ONNX model session."""
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess image to [0,512] range as specified."""
        img = Image.open(image_path).convert('RGB')
        # Resize to 512x512
        img = img.resize((512, 512))
        # Convert to numpy array with [0,512] range
        img_array = np.array(img, dtype=np.float32)
        # Scale from [0,255] to [0,512]
        img_array = img_array * (512.0 / 255.0)
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
