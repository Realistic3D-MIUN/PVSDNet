import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import helperFunctions as helper
import depth_only_parameters as params

class TRTEngine:
    def __init__(self, engine_path, height, width):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.height = height
        self.width = width

        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Status: Engine file not found at {engine_path}.")
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []

        target_input_shape = (1, 3, height, width)

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, target_input_shape)

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            self.context.set_tensor_address(tensor_name, int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                self.host_inputs.append(host_mem)
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                self.host_outputs.append(host_mem)

    def infer(self, input_image_numpy):
        np.copyto(self.inputs[0]['host'], input_image_numpy.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(1, 1, self.height, self.width)

def main():
    
    RESOLUTIONS = [
        (672, 1008),
        (576, 864),
        (448, 672),
        (416, 624),
        (384, 576),
        (288, 432),
        (224, 336)
    ]
    
    params_height = 4032
    params_width = 6048
    INPUT_FOLDER = "./samples/ETH3D/"
    OUTPUT_LOCATION = './output/ETH3D/'
    ENGINE_DIR = "./TRT_Engine/ETH3D"
    
    os.makedirs(OUTPUT_LOCATION, exist_ok=True)
    
    engines = []
    print("Status: Loading TensorRT engines...")
    for h, w in RESOLUTIONS:
        res_str = f"{h}x{w}"
        engine_path = os.path.join(ENGINE_DIR, f"depth_model_{res_str}_fp16.engine")
        if not os.path.exists(engine_path):
            print(f"Error: Engine not found at {engine_path}. Please run export_and_build_trt_eth3d.py first.")
            return
        engines.append(TRTEngine(engine_path, h, w))
    print(f"Status: {len(engines)} engines loaded successfully.")

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return

    transforms_list = [transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()]) for h, w in RESOLUTIONS]

    for filename in image_files:
        print(f"Processing {filename}...")
        file_path = os.path.join(INPUT_FOLDER, filename)
        img_input = Image.open(file_path).convert('RGB')
        
        depth_outputs = []
        
        start_time = time.time()
        
        for i, (engine, trans) in enumerate(zip(engines, transforms_list)):
            
            img_tensor = trans(img_input).unsqueeze(0).numpy()
            depth_out_np = engine.infer(img_tensor)
            depth_out_torch = torch.from_numpy(depth_out_np).to(params.DEVICE)
            
            depth_rescaled = F.interpolate(depth_out_torch, (params_height, params_width), mode='bilinear', align_corners=False)
            depth_outputs.append(depth_rescaled)
            
        depth_final = torch.stack(depth_outputs).mean(dim=0).squeeze(0).squeeze(0)
        
        end_time = time.time()
        print(f"Inference+Fusion time for {filename}: {end_time - start_time:.4f}s")

        torch.save(depth_final, os.path.join(OUTPUT_LOCATION, filename + "_depth_tensor.pt"))
        
        depth_norm = (depth_final - depth_final.min()) / (depth_final.max() - depth_final.min() + 1e-8)
        save_image(depth_norm, os.path.join(OUTPUT_LOCATION, filename + "_depth.png"))
        
        img_out = depth_norm.cpu().detach().numpy()
        img_out_colored = plt.get_cmap('inferno')(img_out)[:, :, :3]
        img_out_colored = (img_out_colored * 255).astype(np.uint8)
        img_out_colored_pil = Image.fromarray(img_out_colored)
        img_out_colored_pil.save(os.path.join(OUTPUT_LOCATION, filename + "_depth_color.png"))

    print(f"\nAll images processed. Results saved to {OUTPUT_LOCATION}")

if __name__ == "__main__":
    main()
