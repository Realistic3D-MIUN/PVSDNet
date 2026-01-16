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
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F

params_height = 4032
params_width = 6048
DEBUG = False

FILE_PATH = "./samples/ETH3D/DSC_0251.JPG"
MODEL_LOCATION = "./checkpoint/depth_only_model.pth"
OUTPUT_LOCATION = './output/ETH3D/'

os.makedirs(OUTPUT_LOCATION, exist_ok=True)
FILENAME = FILE_PATH.split("/")[-1]

model = PVSDNet(total_image_input=params.params_number_input)
model = helper.load_Checkpoint(MODEL_LOCATION, model, load_cpu=True)
model.to(params.DEVICE)
model.eval()

transform_1 = transforms.Compose([transforms.Resize((672, 1008)), transforms.ToTensor()])
transform_2 = transforms.Compose([transforms.Resize((576, 864)), transforms.ToTensor()])
transform_3 = transforms.Compose([transforms.Resize((448, 672)), transforms.ToTensor()])
transform_4 = transforms.Compose([transforms.Resize((416, 624)), transforms.ToTensor()])
transform_5 = transforms.Compose([transforms.Resize((384, 576)), transforms.ToTensor()])
transform_6 = transforms.Compose([transforms.Resize((288, 432)), transforms.ToTensor()])
transform_7 = transforms.Compose([transforms.Resize((224, 336)), transforms.ToTensor()])

img_input = Image.open(FILE_PATH).convert('RGB')

img_input_1 = transform_1(img_input)
img_input_2 = transform_2(img_input)                
img_input_3 = transform_3(img_input)
img_input_4 = transform_4(img_input)
img_input_5 = transform_5(img_input)
img_input_6 = transform_6(img_input)
img_input_7 = transform_7(img_input)

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

img_input_6 = img_input_6.unsqueeze(0).to(params.DEVICE)
depth_out_6 = model(img_input_6)
depth_out_6 = depth_out_6.squeeze(0)
del img_input_6

img_input_7 = img_input_7.unsqueeze(0).to(params.DEVICE)
depth_out_7 = model(img_input_7)
depth_out_7 = depth_out_7.squeeze(0)
del img_input_7

depth_out_1 = F.interpolate(depth_out_1[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_2 = F.interpolate(depth_out_2[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_3 = F.interpolate(depth_out_3[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_4 = F.interpolate(depth_out_4[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_5 = F.interpolate(depth_out_5[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_6 = F.interpolate(depth_out_6[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]
depth_out_7 = F.interpolate(depth_out_7[None], (params_height, params_width), mode='bilinear', align_corners=False)[0, 0]

if DEBUG:
    print(depth_out_1.size())
    print(depth_out_2.size())
    print(depth_out_3.size())
    print(depth_out_4.size())
    print(depth_out_5.size())
    print(depth_out_6.size())
    print(depth_out_7.size())

depth_final = (depth_out_1+depth_out_2+depth_out_3+depth_out_4+depth_out_5+depth_out_6+depth_out_7)/7
torch.save(depth_final,OUTPUT_LOCATION+FILENAME+"_depth_tensor.pt")

depth_final = (depth_final-depth_final.min())/(depth_final.max()-depth_final.min())
save_image(depth_final,OUTPUT_LOCATION+FILENAME+"_depth.png")

img_out = depth_final.squeeze().cpu().detach().numpy()
img_out_colored = plt.get_cmap('inferno')(img_out / np.max(img_out))[:, :, :3]
img_out_colored = (img_out_colored * 255).astype(np.uint8)
img_out_colored = Image.fromarray(img_out_colored)
im3 = img_out_colored

im3.save(OUTPUT_LOCATION+FILENAME+"_depth_color.png")