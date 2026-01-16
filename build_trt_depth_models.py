import os
import subprocess
import torch
import torch.onnx
from models.depth_only_model import PVSDNet
from models.depth_only_lite_model import PVSDNet_Lite
from depth_only_parameters import params_height, params_width, params_number_input, DEVICE
import helperFunctions as helper

# Configuration
ONNX_DIR = "./checkpoint_onnx/depth_only"
ENGINE_DIR = "./TRT_Engine"
CHECKPOINT_STD = "./checkpoint/depth_only_model.pth"
CHECKPOINT_LITE = "./checkpoint/depth_only_lite_model.pth"

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(ENGINE_DIR, exist_ok=True)

def export_to_onnx(model_class, checkpoint_path, output_name):
    print(f"\nStatus: Loading {output_name} Model from {checkpoint_path}")
    model = model_class(total_image_input=params_number_input)
    model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    model.eval()
    model.cuda()
    print(f"Status: {output_name} Model Loaded")

    dummy_img = torch.randn(1, 3, params_height, params_width).cuda()

    onnx_path = os.path.join(ONNX_DIR, f"{output_name}.onnx")
    print(f"Status: Exporting ONNX to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_img,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Status: ONNX exported to {onnx_path}")
    return onnx_path

def build_engine(onnx_path, engine_name):
    # Output engine file pattern: 
    # regular: ./TRT_Engine/depth_only_model_fp16.engine
    # lite: ./TRT_Engine/depth_only_lite_model_fp16.engine
    engine_path = os.path.join(ENGINE_DIR, f"{engine_name}.engine")
    
    if os.path.exists(engine_path):
        print(f"Status: Engine {engine_path} already exists. Skipping build.")
        return

    print(f"Status: Building TensorRT engine for {engine_name}...")
    try:
        subprocess.run([
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16"
        ], check=True)
        print(f"Status: TensorRT engine saved to {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error building engine: {e}")
    except FileNotFoundError:
        print("Error: 'trtexec' not found in PATH. Please make sure TensorRT is installed.")

def main():
    # 1. Process Depth Only (Standard)
    # Target name to match existing convention: depth_only_model_fp16.engine
    onnx_std = export_to_onnx(PVSDNet, CHECKPOINT_STD, "depth_only_model")
    build_engine(onnx_std, "depth_only_model_fp16")

    # 2. Process Depth Only (Lite)
    # Target name to match existing convention: depth_only_lite_model_fp16.engine
    onnx_lite = export_to_onnx(PVSDNet_Lite, CHECKPOINT_LITE, "depth_only_lite_model")
    build_engine(onnx_lite, "depth_only_lite_model_fp16")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
