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
import cv2
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
     (720, 1280),
     (576, 1024),
     (432, 768),
     (288, 512),
     (144, 256)
    ]

    params_height = 1080
    params_width = 1920
    INPUT_FOLDER = "./samples/video_1080p/"
    OUTPUT_LOCATION = './output/Depth_Video/'
    ENGINE_DIR = "./TRT_Engine/1080p"
    
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

    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"No videos found in {INPUT_FOLDER}")
        return

    transforms_list = [transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()]) for h, w in RESOLUTIONS]

    for video_name in video_files:
        video_path = os.path.join(INPUT_FOLDER, video_name)
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None: 
            fps = 30
        
        output_filename = os.path.splitext(video_name)[0] + "_depth_1080p.mp4"
        output_path = os.path.join(OUTPUT_LOCATION, output_filename)
        
        # Force 1080p output for VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (params_width, params_height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_name} ({total_frames} frames, Resizing to {params_width}x{params_height} for output)")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enforce 1080p input for the fusion logic (resizes to params_width x params_height)
            if frame.shape[1] != params_width or frame.shape[0] != params_height:
                frame = cv2.resize(frame, (params_width, params_height))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_input = Image.fromarray(frame_rgb)
            
            depth_outputs = []
            start_time = time.time()
            
            for i, (engine, trans) in enumerate(zip(engines, transforms_list)):
                img_tensor = trans(img_input).unsqueeze(0).numpy()
                depth_out_np = engine.infer(img_tensor)
                depth_out_torch = torch.from_numpy(depth_out_np).to(params.DEVICE)
                
                # Rescale each resolution back to the target 1080p resolution
                depth_rescaled = F.interpolate(depth_out_torch, (params_height, params_width), mode='bilinear', align_corners=False)
                depth_outputs.append(depth_rescaled)
                
            # Multi-resolution fusion (averaging)
            depth_final = torch.stack(depth_outputs).mean(dim=0).squeeze(0).squeeze(0)
            
            # Normalization
            depth_min = depth_final.min()
            depth_max = depth_final.max()
            depth_norm = (depth_final - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Color mapping (using inferno as requested)
            img_out = depth_norm.cpu().detach().numpy()
            img_out_colored = plt.get_cmap('inferno')(img_out)[:, :, :3]
            img_out_colored = (img_out_colored * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV VideoWriter
            img_out_bgr = cv2.cvtColor(img_out_colored, cv2.COLOR_RGB2BGR)
            video_writer.write(img_out_bgr)
            
            end_time = time.time()
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}/{total_frames} - Inference time: {end_time - start_time:.4f}s", end="\r")

        cap.release()
        video_writer.release()
        print(f"Finished processing {video_name}. Result saved to {output_path}")

    print(f"\nAll videos processed. Results saved to {OUTPUT_LOCATION}")

if __name__ == "__main__":
    main()
