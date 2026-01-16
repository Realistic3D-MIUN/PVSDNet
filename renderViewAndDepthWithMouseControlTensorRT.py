import os
import argparse
import time
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as FF
from torchvision import transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import parameters_pvsdnet as params

class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print("Status: Loading TensorRT engine from: ", engine_path)
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Status: Engine file not found at {engine_path}.")
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.inputs = {}  # Dict to store input bindings by name
        self.outputs = {} # Dict to store output bindings by name
        
        # Iterate over all tensors in the engine
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            
            # We assume static shapes for this specific use case as per the export script
            # If dynamic shapes were used, set_input_shape would be needed here.
            
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            
            print(f"Tensor: {tensor_name}, Shape: {shape}, Size: {size}, Dtype: {dtype}, Mode: {mode}")

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to bindings list (order matters for execute_async_v2 if used, 
            # but v3 uses named binding or address setting. We use address setting here.)
            self.bindings.append(int(device_mem))
            self.context.set_tensor_address(tensor_name, int(device_mem))

            binding_data = {
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype,
                'size': size
            }

            if mode == trt.TensorIOMode.INPUT:
                self.inputs[tensor_name] = binding_data
            else:
                self.outputs[tensor_name] = binding_data

    def infer(self, feed_dict):
        """
        feed_dict: dictionary mapping input tensor names to numpy arrays
        """
        # Copy inputs to host memory and then to device
        for name, data in feed_dict.items():
            if name not in self.inputs:
                raise ValueError(f"Input name {name} not found in engine inputs.")
            
            target_dtype = self.inputs[name]['dtype']
            # Ensure data is flat and of correct type
            data_flat = np.ascontiguousarray(data.flatten().astype(target_dtype))
            
            expected_size = self.inputs[name]['size']
            if data_flat.size != expected_size:
                raise ValueError(f"Input {name} size mismatch. Expected {expected_size}, got {data_flat.size}")
            
            np.copyto(self.inputs[name]['host'], data_flat)
            cuda.memcpy_htod_async(self.inputs[name]['device'], self.inputs[name]['host'], self.stream)

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back to host
        for name in self.outputs:
            cuda.memcpy_dtoh_async(self.outputs[name]['host'], self.outputs[name]['device'], self.stream)
        
        # Synchronize
        self.stream.synchronize()

        # Return a dictionary of outputs
        results = {}
        for name, data in self.outputs.items():
            results[name] = data['host'].reshape(data['shape'])
        
        return results

# Argument Parsing
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=params.params_height)
parser.add_argument('--width', type=int, default=params.params_width)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity.")
parser.add_argument('--model_type', type=str, default='pvsdnet', choices=['pvsdnet', 'pvsdnet_lite'], help="Choose the model type")
parser.add_argument('--engine_path', type=str, default=None, help="Explicit path to engine. If None, defaults will be used based on model_type.")
parser.add_argument('--input_image', type=str, default="./samples/PVSDNet_Samples/person.jpeg", help="Input image path")

opt, _ = parser.parse_known_args()

# Determine Engine Path
if opt.engine_path is None:
    if opt.model_type == 'pvsdnet_lite':
        opt.engine_path = "./TRT_Engine/pvsdnet/pvsdnet_lite_fp16.engine"
    else:
        opt.engine_path = "./TRT_Engine/pvsdnet/pvsdnet_fp16.engine"

# GUI Setup
root = tk.Tk()
root.title(f"PVSDNet Renderer (TensorRT) - {opt.model_type}")
SCALE = opt.scale

# Load Engine
print("Status: Loading TensorRT Engine...")
try:
    engine = TRTEngine(opt.engine_path)
    print("Status: TensorRT Engine Loaded!")
except Exception as e:
    print(f"Error loading engine: {e}")
    exit(1)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((opt.height, opt.width)),
    transforms.ToTensor()
])

if not os.path.exists(opt.input_image):
    print(f"Error: Input image not found at {opt.input_image}")
    exit(1)

try:
    img_input_pil = Image.open(opt.input_image).convert('RGB')
    img_input_tensor = transform(img_input_pil).unsqueeze(0) # (1, 3, H, W)
    img_input_numpy = img_input_tensor.numpy()
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

def getPositionVector(x, y, z=0):
    vector = np.zeros((1, 3), dtype=np.float32)
    normalized_x = (float(format(x, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_y = (float(format(y, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_z = (float(format(z, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    vector[0][0] = normalized_x
    vector[0][1] = normalized_y
    vector[0][2] = normalized_z
    return vector

def stack_images_side_by_side(im1, im1_d, im1_c):
    if im1.height != im1_d.height:
        im1_d = im1_d.resize((im1_d.width, im1.height))
        im1_c = im1_c.resize((im1_c.width, im1.height))
    
    combined_width = im1.width + im1_d.width + im1_c.width
    combined_height = im1.height
    combined_image = Image.new("RGB", (combined_width, combined_height))
    
    combined_image.paste(im1, (0, 0))
    combined_image.paste(im1_d, (im1.width, 0))
    combined_image.paste(im1_c, (im1.width + im1_d.width, 0))
    
    return combined_image

def renderSingleFrame(x, y):
    pos_numpy = getPositionVector(x, 0, y)
    
    feed_dict = {
        'input_image': img_input_numpy,
        'input_pos': pos_numpy
    }

    start_time = time.time()
    results = engine.infer(feed_dict)
    end_time = time.time()
    
    print(f"x={x:.4f}, y={y:.4f} | FPS: {1/(end_time-start_time):.2f} | Inf Time: {end_time-start_time:.4f}")

    # Process View Synthesis Output
    predicted_img_np = results['output_image'] # (1, 3, H, W)
    im = FF.to_pil_image(torch.from_numpy(predicted_img_np[0]))
    newsize = (opt.width*SCALE, opt.height*SCALE)
    im1 = im.resize(newsize)

    # Process Depth Output
    predicted_depth_np = results['output_depth'] # (1, 1, H, W) or similar
    depth_tensor = torch.from_numpy(predicted_depth_np[0])
    
    # Normalize depth
    d_min = depth_tensor.min()
    d_max = depth_tensor.max()
    if d_max - d_min > 1e-6:
        depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
    else:
        depth_tensor = torch.zeros_like(depth_tensor)

    im_d = FF.to_pil_image(depth_tensor)
    im1_d = im_d.resize(newsize)

    # Colorized Depth
    img_out = depth_tensor.squeeze().cpu().numpy()
    img_out_colored = plt.get_cmap('inferno')(img_out)[:, :, :3]
    img_out_colored = (img_out_colored * 255).astype(np.uint8)
    img_c = Image.fromarray(img_out_colored)
    img_c = img_c.resize(newsize)

    combined_img = stack_images_side_by_side(im1, im1_d, img_c)
    return combined_img

def update_image(x, y):
    # Clamp values
    limit = 0.1
    x = max(-limit, min(limit, x))
    y = max(-limit, min(limit, y))
    
    img = renderSingleFrame(x, y)
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

# Initial render
print("Status: Performing initial render...")
initial_img = renderSingleFrame(0, 0)
img_tk = ImageTk.PhotoImage(initial_img)

# Label
label = tk.Label(root, image=img_tk)
label.image = img_tk
label.pack()

# Mouse Control
def on_mouse_drag(event):
    x_offset = (event.x - root.winfo_width() / 2) / opt.mouse_sensitivity
    y_offset = (event.y - root.winfo_height() / 2) / opt.mouse_sensitivity
    update_image(x_offset, y_offset)

root.bind('<B1-Motion>', on_mouse_drag)

print("Status: Application successfully started.")
root.mainloop()
