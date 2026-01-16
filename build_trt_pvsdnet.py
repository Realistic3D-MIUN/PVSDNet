import os
import subprocess
import torch
import torch.onnx
from models.pvsdnet_model import PVSDNet
from models.pvsdnet_lite_model import PVSDNet_Lite
import parameters_pvsdnet as params
import helperFunctions as helper

# Configuration
ONNX_DIR = "./checkpoint_onnx/pvsdnet"
ENGINE_DIR = "./TRT_Engine/pvsdnet"
CHECKPOINT_STD = "./checkpoint/pvsdnet_model.pth"
CHECKPOINT_LITE = "./checkpoint/pvsdnet_lite_model.pth"

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(ENGINE_DIR, exist_ok=True)

def export_to_onnx(model_class, checkpoint_path, output_name):
    print(f"\nStatus: Loading {output_name} Model from {checkpoint_path}")
    model = model_class(total_image_input=params.params_number_input)
    model = helper.load_Checkpoint(checkpoint_path, model, load_cpu=True)
    model.eval()
    model.cuda()
    print(f"Status: {output_name} Model Loaded")

    dummy_img = torch.randn(1, 3, params.params_height, params.params_width).cuda()
    # Position vector is (1, 3)
    dummy_pos = torch.randn(1, 3).cuda()

    onnx_path = os.path.join(ONNX_DIR, f"{output_name}.onnx")
    print(f"Status: Exporting ONNX to {onnx_path}...")
    
    torch.onnx.export(
        model,
        (dummy_img, dummy_pos),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_image', 'input_pos'],
        output_names=['output_image', 'output_depth']
    )
    print(f"Status: ONNX exported to {onnx_path}")
    return onnx_path

def build_engine(onnx_path, engine_name):
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
    # 1. Process PVSDNet (Standard)
    onnx_std = export_to_onnx(PVSDNet, CHECKPOINT_STD, "pvsdnet")
    build_engine(onnx_std, "pvsdnet_fp16")

    # 2. Process PVSDNet (Lite)
    onnx_lite = export_to_onnx(PVSDNet_Lite, CHECKPOINT_LITE, "pvsdnet_lite")
    build_engine(onnx_lite, "pvsdnet_lite_fp16")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
