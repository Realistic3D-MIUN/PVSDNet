import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import depth_only_parameters as params
import numpy as np
import torch
from models.depth_only_model import PVSDNet
from torchvision.utils import save_image
import helperFunctions as helper
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F

params_height = 450
params_width = 600
DEBUG = False

FILE_PATH = "./samples/NYU/0000000048.png"
MODEL_LOCATION = "./checkpoint/depth_only_model.pth"
OUTPUT_LOCATION = './output/NYU/'

os.makedirs(OUTPUT_LOCATION, exist_ok=True)
FILENAME = FILE_PATH.split("/")[-1]

model = PVSDNet(total_image_input=params.params_number_input)
model = helper.load_Checkpoint(MODEL_LOCATION, model, load_cpu=True)
model.to(params.DEVICE)
model.eval()

transform_1 = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
transform_2 = transforms.Compose([transforms.Resize((432, 576)), transforms.ToTensor()])
transform_3 = transforms.Compose([transforms.Resize((384, 512)), transforms.ToTensor()])
transform_4 = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])
transform_5 = transforms.Compose([transforms.Resize((192, 256)), transforms.ToTensor()])

img_input = Image.open(FILE_PATH).convert('RGB')

img_input_1 = transform_1(img_input)
img_input_2 = transform_2(img_input)                
img_input_3 = transform_3(img_input)
img_input_4 = transform_4(img_input)
img_input_5 = transform_5(img_input)

img_input_1 = img_input_1.unsqueeze(0).to(params.DEVICE)
depth_out_1 = model(img_input_1)
depth_out_1 = depth_out_1.squeeze(0)
del img_input_1

img_input_2 = img_input_2.unsqueeze(0).to(params.DEVICE)
depth_out_2 = model(img_input_2)
depth_out_2 = depth_out_2.squeeze(0)
del img_input_2

img_input_3 = img_input_3.unsqueeze(0).to(params.DEVICE)
depth_out_3 = model(img_input_3)
depth_out_3 = depth_out_3.squeeze(0)
del img_input_3

img_input_4 = img_input_4.unsqueeze(0).to(params.DEVICE)
depth_out_4 = model(img_input_4)
depth_out_4 = depth_out_4.squeeze(0)
del img_input_4

img_input_5 = img_input_5.unsqueeze(0).to(params.DEVICE)
depth_out_5 = model(img_input_5)
depth_out_5 = depth_out_5.squeeze(0)
del img_input_5

depth_out_1 = F.interpolate(depth_out_1[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_2 = F.interpolate(depth_out_2[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_3 = F.interpolate(depth_out_3[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_4 = F.interpolate(depth_out_4[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_5 = F.interpolate(depth_out_5[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]

if DEBUG:
    print(depth_out_1.size())
    print(depth_out_2.size())
    print(depth_out_3.size())
    print(depth_out_4.size())
    print(depth_out_5.size())

depth_final = (depth_out_1+depth_out_2+depth_out_3+depth_out_4+depth_out_5)/5
torch.save(depth_final,OUTPUT_LOCATION+FILENAME+"_depth_tensor.pt")

depth_final = (depth_final-depth_final.min())/(depth_final.max()-depth_final.min())
save_image(depth_final,OUTPUT_LOCATION+FILENAME+"_depth.png")

img_out = depth_final.squeeze().cpu().detach().numpy()
img_out_colored = plt.get_cmap('inferno')(img_out / np.max(img_out))[:, :, :3]
img_out_colored = (img_out_colored * 255).astype(np.uint8)
img_out_colored = Image.fromarray(img_out_colored)
im3 = img_out_colored

im3.save(OUTPUT_LOCATION+FILENAME+"_depth_color.png")