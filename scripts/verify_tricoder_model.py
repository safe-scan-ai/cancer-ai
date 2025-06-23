import onnxruntime as ort
import numpy as np

def verify_onnx_model(model_path):
    # Create ONNX Runtime session
    session = ort.InferenceSession(model_path)
    
    # Get input and output details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    
    print(f"Input name: {input_name}, shape: {input_shape}")
    print(f"Output name: {output_name}, shape: {output_shape}")
    
    # Create a dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    outputs = session.run([output_name], {input_name: dummy_input})
    
    # Check output shape and values
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output sum: {outputs[0].sum()}")
    print(f"Output min: {outputs[0].min()}, max: {outputs[0].max()}")

if __name__ == "__main__":
    verify_onnx_model("sample_models/sample_tricorder_model.onnx")
