import onnxruntime
import numpy as np
from PIL import Image

model_path = "best_model.onnx"
image_path = "image.jpg"
target_size = (512, 512)

try:
    session = onnxruntime.InferenceSession(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Load and preprocess the image
img = Image.open(image_path)
img = img.resize(target_size)  # Resize the image
img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize and convert to float32

# Ensure 3 channels (RGB) if image is grayscale
if img_array.shape[-1] != 3:
    img_array = np.stack((img_array,) * 3, axis=-1)

# Transpose image to (C, H, W) format
img_array = np.transpose(img_array, (2, 0, 1))

# Add batch dimension
input_batch = np.expand_dims(img_array, axis=0)

# Prepare input dictionary for the model
input_name = session.get_inputs()[0].name
input_data = {input_name: input_batch}

# Run inference
try:
    results = session.run(None, input_data)[0]
    print(results)
except Exception as e:
    print(f"Failed to run model inference: {e}")
    exit(1)