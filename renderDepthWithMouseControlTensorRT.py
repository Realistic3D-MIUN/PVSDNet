import os
from pathlib import Path
import argparse
import time
import math
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms import functional as FF
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from depth_only_parameters import *
import depth_only_parameters as params

class TRTEngine:
    def __init__(self, engine_path, height, width):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print("Status: Loading TensorRT engine from: ",engine_path)
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
                print(f"Setting input shape for {tensor_name} to {target_input_shape}")
                try:
                    self.context.set_input_shape(tensor_name, target_input_shape)
                except Exception as e:
                    print("Failed to set input shape to ",target_input_shape)
                    print("TensorRT engine shape is different from your passed input.")
                    raise e

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)

            shape = self.context.get_tensor_shape(tensor_name)

            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            print(f"Tensor: {tensor_name}, Resolved Shape: {shape}, Size: {size}, Dtype: {dtype}")

            if size <= 0:
                raise ValueError(f"Invalid tensor size {size} for {tensor_name}. Shape resolved to {shape}.")

            try:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            except Exception as e:
                raise RuntimeError(f"Failed to allocate memory for {tensor_name} (size: {size} elements): {e}")
            
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
        self.stream.synchronize() # this is for accurate timing
        return self.outputs[0]['host']

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=params.params_height)
parser.add_argument('--width', type=int, default=params.params_width)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity for small baseline network to limit going over the baseline.")
parser.add_argument('--model_type', type=str, default='regular', choices=['regular', 'lite'], help="Choose the model type: 'regular' or 'lite'")
parser.add_argument('--engine_path', type=str, default=None, help="Explicit path to the TensorRT engine file. If not set, defaults are used based on model_type.")
parser.add_argument('--input_image', type=str, default="./samples/Wild/plant.jpeg", help="Location of input imnage for which novel views are needed to be synthesized")
parser.add_argument('--color', action='store_true', help='Whether to perform color change in depth or not')
opt, _ = parser.parse_known_args()

# Determine Engine Path if not provided
if opt.engine_path is None:
    if opt.model_type == 'lite':
        opt.engine_path = "./TRT_Engine/depth_only_lite_model_fp16.engine"
    else:
        opt.engine_path = "./TRT_Engine/depth_only_model_fp16.engine"



root = tk.Tk()
root.title(f"PVSDNet-Depth-Only Renderer (TensorRT) - {opt.model_type}")
SCALE = opt.scale

print("Status: Loading TensorRT Engine")
try:
    engine = TRTEngine(opt.engine_path, opt.height, opt.width)
    print("Status: TensorRT Engine Loaded!")
except Exception as e:
    print("Error loading engine: ",e)
    exit(1)

transform = transforms.Compose([transforms.Resize((opt.height, opt.width)),
                                    transforms.ToTensor()])

if not os.path.exists(opt.input_image):
    print(f"Warning: Input image not found at {opt.input_image}. Please check path.")

try:
    img_input_pil = Image.open(opt.input_image).convert('RGB')
    img_input_tensor = transform(img_input_pil).unsqueeze(0)
    img_input_numpy = img_input_tensor.numpy()

except Exception as e:
    print("Error loading image: ",e)
    exit(1)

def getPositionVector(x,y,z=0):
    vector = torch.zeros((1,3),dtype=torch.float)
    normalized_x = (float(format(x,'.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_y = (float(format(y,'.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_z = (float(format(z,'.7f')) - (-0.1)) / (0.1 - (-0.1))
    vector[0][0] = normalized_x
    vector[0][1] = normalized_y
    vector[0][2] = normalized_z
    return vector

def stack_images_side_by_side(im1, im1_d, im1_c):
    if im1.height != im1_d.height:
        raise ValueError("The images must have the same height to stack")
    
    combined_width = im1.width + im1_d.width + im1_c.width
    combined_height = im1.height
    combined_image = Image.new("RGB", (combined_width, combined_height))
    
    combined_image.paste(im1, (0, 0))  
    combined_image.paste(im1_d, (im1.width, 0))
    combined_image.paste(im1_c, (im1.width+im1_d.width, 0))
    
    return combined_image

def renderSingleFrame(x,y):

    start_time = time.time()
    # Inference with TensorRT
    output_flat = engine.infer(img_input_numpy) # GPU sync is handled inside infer() via stream.synchronize(), no need for torch.cuda.synchronize()
    end_time = time.time()

    print(f"FPS: {1/(end_time-start_time):.2f} | Inf Time: {end_time-start_time:.4f}")
    
    output_shape = (1, 1, opt.height, opt.width)
    predicted_img_np = output_flat.reshape(output_shape)
    
    output_shape = (1, 1, opt.height, opt.width)
    predicted_img_np = output_flat.reshape(output_shape)
    
    img_data = predicted_img_np[0]

    min_val = np.min(img_data)
    max_val = np.max(img_data)
    
    if max_val - min_val > 1e-5:
        img_data = (img_data - min_val) / (max_val - min_val)
    else:
        img_data = np.zeros_like(img_data)

    im = FF.to_pil_image(torch.from_numpy(img_data)) 

    img_for_plot = img_data.squeeze()
    img_out_colored = plt.get_cmap('inferno')(img_for_plot)[:, :, :3]
    img_out_colored = (img_out_colored * 255).astype(np.uint8)
    img_out_colored = Image.fromarray(img_out_colored)
    im3 = img_out_colored
    
    newsize = (opt.width*SCALE, opt.height*SCALE)
    im1 = im.resize(newsize)

    im2 = FF.to_pil_image(img_input_tensor[0])
    im2 = im2.resize(newsize)
    im3 = im3.resize(newsize)

    combined_image = stack_images_side_by_side(im2,im1,im3)
    return combined_image

def update_image(x,y):
    img = renderSingleFrame(x,y)
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

initial_img = renderSingleFrame(0,0)
print("Got initial image: ",type(initial_img))
img_tk = ImageTk.PhotoImage(initial_img)

label = tk.Label(root, image=img_tk)
label.image = img_tk
label.pack()

# Infer new depth each time mouse is dragged (just to test inference speed)
def on_mouse_drag(event):
    x_offset = (event.x - root.winfo_width() / 2) / opt.mouse_sensitivity
    y_offset = (event.y - root.winfo_height() / 2) / opt.mouse_sensitivity
    update_image(x_offset, y_offset)

root.bind('<B1-Motion>', on_mouse_drag)

root.mainloop()