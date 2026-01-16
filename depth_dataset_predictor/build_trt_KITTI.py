import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.onnx
import subprocess
from models.depth_only_model import PVSDNet
import depth_only_parameters as params
import helperFunctions as helper

RESOLUTIONS = [
    (352, 1216),
    (320, 1104),
    (208, 720),
    (176, 608)
]

MODEL_LOCATION = "./checkpoint/depth_only_model.pth"
ONNX_DIR = "./checkpoint_onnx/KITTI"
ENGINE_DIR = "./TRT_Engine/KITTI"

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(ENGINE_DIR, exist_ok=True)

print("Status: Loading PVSDNet Model")
model = PVSDNet(total_image_input=params.params_number_input)
model = helper.load_Checkpoint(MODEL_LOCATION, model, load_cpu=True)
model.eval()
model.cuda()
print("Status: PVSDNet Model Loaded")

for i, (h, w) in enumerate(RESOLUTIONS):
    res_str = f"{h}x{w}"
    onnx_path = os.path.join(ONNX_DIR, f"depth_model_{res_str}.onnx")
    engine_path = os.path.join(ENGINE_DIR, f"depth_model_{res_str}_fp16.engine")
    
    print(f"\n--- Processing Resolution {i+1}/4: {res_str} ---")

    if not os.path.exists(onnx_path):
        print(f"Status: Exporting ONNX for {res_str}...")
        dummy_input = torch.randn(1, 3, h, w).cuda()
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,        
            opset_version=13,          
            do_constant_folding=True,  
            input_names=['input'],     
            output_names=['output']    
        )
        print(f"Status: ONNX exported to {onnx_path}")
    else:
        print(f"Status: ONNX for {res_str} already exists.")

    if not os.path.exists(engine_path):
        print(f"Status: Building TensorRT engine for {res_str} (this may take a few minutes)...")
        try:
            subprocess.run([
                "trtexec", 
                f"--onnx={onnx_path}", 
                f"--saveEngine={engine_path}", 
                "--fp16"
            ], check=True)
            print(f"Status: TensorRT engine saved to {engine_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error building engine for {res_str}: {e}")
        except FileNotFoundError:
            print("Error: 'trtexec' not found in PATH. Please make sure TensorRT is installed and trtexec is in your PATH.")
    else:
        print(f"Status: Engine for {res_str} already exists.")

print("\nAll engines processed.")
