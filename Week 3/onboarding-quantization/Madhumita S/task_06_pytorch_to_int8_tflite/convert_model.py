import torch
import onnx
import tensorflow as tf
import numpy as np

from onnx2tf import convert

from model_definition import SimpleCNN

#Loading the model
model = SimpleCNN()
try:
    #Loading Weights
    model.load_state_dict(torch.load(".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\model.pth",
                                     map_location="cpu"))
except FileNotFoundError:
    print("File not found")
#Setting evaluation Mode
model.eval()

# Create Dummy input
dummy_input = torch.randn(1,1,28,28) #(batch_size, channel,height,width)
# creates 1 image with 1 channel and has 28x28 pixels

try:
    # Exporting to onnx
    torch.onnx.export(model, dummy_input,
                      ".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\model.onnx",
                      export_params=True,opset_version=13,do_constant_folding=True,
                      input_names=["input"],output_names=["output"])
except Exception as e:
    print(e)

onnx_model = onnx.load(".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\model.onnx")
try:
    onnx.checker.check_model(onnx_model)
    print("Onnx model is checked")
    print("Inputs")
    for inp in onnx_model.graph.input:
        print(inp.name)
    print("Outputs")
    for out in onnx_model.graph.output:
        print(out.name)
except Exception :
    print("Onnx model is not valid")
try:
    calib_data =[]
    convert(input_onnx_file_path = ".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\model.onnx",
        output_folder_path = ".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\tf_model",
        flatbuffer_direct_output_saved_model = True)
    print("Model converted and saved into tf_model")
except Exception as e:
    import traceback
    print(e)
    traceback.print_exc()
    print("Conversion failed")

