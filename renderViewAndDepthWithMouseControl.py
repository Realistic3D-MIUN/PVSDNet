# System
import os
from pathlib import Path
import argparse
import time
import math
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from models.pvsdnet_model import PVSDNet
from models.pvsdnet_lite_model import PVSDNet_Lite
import helperFunctions as helper
import parameters_pvsdnet as params
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=params.params_height)
parser.add_argument('--width', type=int, default=params.params_width)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity for small baseline network to limit going over the baseline.")
parser.add_argument('--model_type', type=str, default='pvsdnet', choices=['pvsdnet', 'pvsdnet_lite'], help="Choose the model type between 'pvsdnet' and 'pvsdnet_lite'")
parser.add_argument('--checkpoint', type=str, default="./checkpoint/pvsdnet_model.pth", help="Checkpoint path for pvsdnet")
parser.add_argument('--checkpoint_lite', type=str, default="./checkpoint/pvsdnet_lite_model.pth", help="Checkpoint path for pvsdnet_lite")
parser.add_argument('--input_image', type=str, default="./samples/PVSDNet_Samples/person.jpeg", help="Location of input imnage for which novel views are needed to be synthesized")

opt, _ = parser.parse_known_args()

# Initialize the main application window
root = tk.Tk()
root.title("PVSDNet Renderer")
SCALE = opt.scale

# Load pre-processed data
if opt.model_type == 'pvsdnet_lite':
    model = PVSDNet_Lite(total_image_input=params.params_number_input)
    model = helper.load_Checkpoint(opt.checkpoint_lite, model, load_cpu=True)
else:
    model = PVSDNet(total_image_input=params.params_number_input)
    model = helper.load_Checkpoint(opt.checkpoint, model, load_cpu=True)
model.to(params.DEVICE)
model.eval()
print("Status: Model Loaded!")

transform = transforms.Compose([transforms.Resize((opt.height, opt.width)),
                                    transforms.ToTensor()])

img_input = Image.open(opt.input_image).convert('RGB')
img_input = transform(img_input).unsqueeze(0).to(params.DEVICE)

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
    # Ensure both images have the same height
    if im1.height != im1_d.height:
        raise ValueError("The images must have the same height to stack them side by side.")
    
    # Create a new blank image with the combined width of the two images and the same height
    combined_width = im1.width + im1_d.width + im1_c.width
    combined_height = im1.height
    combined_image = Image.new("RGB", (combined_width, combined_height))
    
    # Paste the two images side by side
    combined_image.paste(im1, (0, 0))  # Paste im1 at the left
    combined_image.paste(im1_d, (im1.width, 0))  # Paste im1_d to the right of im1
    combined_image.paste(im1_c, (im1.width+im1_d.width, 0))  # Paste im1_d to the right of im1
    
    return combined_image


def renderSingleFrame(x,y):
    pos = getPositionVector(x,0,y).unsqueeze(0).to(params.DEVICE)
    print(x,y)
    start_time = time.time()
    predicted_img, predicted_depth = model(img_input,pos)
    torch.cuda.synchronize() # uncomment when running on GPU to get accurate timing
    end_time = time.time()
    print("FPS Rate: ",1/(end_time-start_time))
    print("Inference: ",(end_time-start_time))
    im = torchvision.transforms.functional.to_pil_image(predicted_img[0])
    newsize = (opt.width*SCALE, opt.height*SCALE)
    im1 = im.resize(newsize)
    #im1.save("NVS.png")
    

    predicted_depth = (predicted_depth-predicted_depth.min())/(predicted_depth.max()-predicted_depth.min())

    im_d = torchvision.transforms.functional.to_pil_image(predicted_depth[0])
    newsize = (opt.width*SCALE, opt.height*SCALE)
    im1_d = im_d.resize(newsize)

    img_out = predicted_depth.squeeze().cpu().detach().numpy()
    img_out_colored = plt.get_cmap('inferno')(img_out / np.max(img_out))[:, :, :3]
    img_out_colored = (img_out_colored * 255).astype(np.uint8)
    img_out_colored = Image.fromarray(img_out_colored)
    img_c = img_out_colored
    
    #img_c.save("Color.png")

    combined_img = stack_images_side_by_side(im1,im1_d, img_c)
    return combined_img

    #return im1,im1_d

# Function to update the image based on new camera pose
def update_image(x,y):
    if x>0.1:
        x = 0.1
    elif x<-0.1:
        x = -0.1

    if y>0.1:
        y = 0.1
    elif y<-0.1:
        y = -0.1
    
    img = renderSingleFrame(x,y)
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

# Initial render
initial_img = renderSingleFrame(0,0)
#newsize = (opt.width*SCALE,opt.height*SCALE)
#initial_img = initial_img.resize(newsize)
print("Got initial image: ",type(initial_img))
img_tk = ImageTk.PhotoImage(initial_img)

# Label to display the image
label = tk.Label(root, image=img_tk)
label.image = img_tk
label.pack()

# Update the camera pose based on mouse movement
def on_mouse_drag(event):
    x_offset = (event.x - root.winfo_width() / 2) / opt.mouse_sensitivity
    y_offset = (event.y - root.winfo_height() / 2) / opt.mouse_sensitivity
    update_image(x_offset, y_offset)

# Bind mouse drag event to the function
root.bind('<B1-Motion>', on_mouse_drag)

# Start the GUI loop
root.mainloop()

