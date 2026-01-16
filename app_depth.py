import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import depth_only_parameters as params

from models.depth_only_model import PVSDNet
from models.depth_only_lite_model import PVSDNet_Lite

import helperFunctions as helper
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip

local_ip = get_local_ip()
print("Local IP Address:", local_ip)

def get_valid_resolutions(width, height):
    """Dynamically determines valid resolutions based on input size.
    - Caps the highest resolution at 1024px to avoid unnecessary high-res computations.
    - Uses 6 resolutions for large images to improve multi-scale fusion quality.
    - Uses 4 resolutions for smaller images (< 512px width or height).
    """
    def make_divisible(n, base=16):
        return max(base, int(round(n / base) * base))

    max_resolution = 1024
    high_w, high_h = make_divisible(min(width, max_resolution)), make_divisible(min(height, max_resolution))

    # Calculate more intermediate steps for better fusion
    r80_w, r80_h = make_divisible(int(high_w // 1.25)), make_divisible(int(high_h // 1.25))
    r66_w, r66_h = make_divisible(int(high_w // 1.5)), make_divisible(int(high_h // 1.5))
    r50_w, r50_h = make_divisible(int(high_w // 2)), make_divisible(int(high_h // 2))
    r40_w, r40_h = make_divisible(int(high_w // 2.5)), make_divisible(int(high_h // 2.5))
    r33_w, r33_h = make_divisible(max(256, int(high_w // 3))), make_divisible(max(256, int(high_h // 3)))

    if width < 512 or height < 512:
        return [(high_w, high_h), (r80_w, r80_h), (r66_w, r66_h), (r50_w, r50_h)]
    else:
        return [
            (high_w, high_h), 
            (r80_w, r80_h), 
            (r66_w, r66_h), 
            (r50_w, r50_h), 
            (r40_w, r40_h), 
            (r33_w, r33_h)
        ]


def get_transforms(resolutions):
    return [transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()]) for w, h in resolutions]

def get_prediction(image, transform, model):
    img_input = image.convert('RGB')
    img_input = transform(img_input).unsqueeze(0).to(params.DEVICE)
    depth_out = model(img_input).detach().squeeze(0).to("cpu")
    return depth_out

def predict_single_image(image, model_type):
    if image is None:
        return None, None
        
    # Select model class and checkpoint
    if model_type == "Lite":
        model_class = PVSDNet_Lite
        checkpoint = params.MODEL_Small_Location
    else:  # Default to "Large"
        model_class = PVSDNet
        checkpoint = params.MODEL_Large_Location

    model = model_class(total_image_input=params.params_number_input)
    model = helper.load_Checkpoint(checkpoint, model, load_cpu=True)
    model.to(params.DEVICE)
    model.eval()

    original_width, original_height = image.size

    resolutions = get_valid_resolutions(original_width, original_height)
    print(f"Resolutions: {resolutions} for Model Type: {model_type}")
    transforms_list = get_transforms(resolutions)

    depth_maps = [get_prediction(image, t, model) for t in transforms_list]

    depth_maps_resized = [
        F.interpolate(depth[None], (original_height, original_width), mode='bilinear', align_corners=False)[0, 0]
        for depth in depth_maps
    ]

    depth_final = sum(depth_maps_resized) / len(depth_maps_resized)

    depth_image = (depth_final - depth_final.min()) / (depth_final.max() - depth_final.min())

    img_out = depth_image.numpy()
    img_out_colored = plt.get_cmap('inferno')(img_out / np.max(img_out))[:, :, :3]
    img_out_colored = (img_out_colored * 255).astype(np.uint8)

    gray_scale_img_out = (depth_image.numpy() * 255).astype(np.uint8)

    return Image.fromarray(img_out_colored), Image.fromarray(gray_scale_img_out)

with gr.Blocks(title="PVSDNet-Depth-Only Model", theme="default") as demo:
    gr.Markdown(
    """
    ## PVSDNet-Depth-Only ZeroShot Relative Depth Estimation Model
    * Upload an image and get its depth estimation with multi-scale fusion.
    * Images use 2 - 6 resolutions for multi-scale fusion.
    
    **Note:** Huggingface demo is running on CPU so inference speeds will be slow.
    ### Head to our [Project Page](https://realistic3d-miun.github.io/PVSDNet/) for more details about the models.
    """)
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="RGB Image", height=384)
            with gr.Accordion("Advanced Settings", open=False):
                model_type_dropdown = gr.Dropdown(["Large", "Lite"], label="Model Type", value="Large")
            generate_btn = gr.Button("Estimate Depth", variant="primary")
            
        with gr.Column():
            output_color = gr.Image(type="pil", label="Depth Map (Color)", height=384)
            output_gray = gr.Image(type="pil", label="Depth Map (Grayscale)", height=384)

    generate_btn.click(
        fn=predict_single_image,
        inputs=[img_input, model_type_dropdown],
        outputs=[output_color, output_gray]
    )

    gr.Markdown("### Example Samples")
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=2): gr.Markdown("**Example Image (Click to load)**")
            with gr.Column(scale=1): gr.Markdown("**Resolution**")
            with gr.Column(scale=2): gr.Markdown("**Fusion Resolutions**")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                diode_preview = gr.Image("./samples/DIODE/00022_00195_outdoor_010_030.png", label="DIODE", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("1024 x 768")
            with gr.Column(scale=2):
                gr.Markdown("1024x768, 816x608, 688x512, 512x384, 416x304, 336x256")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                eth3d_preview = gr.Image("./samples/ETH3D/DSC_0243.JPG", label="ETH3D", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("6048 x 4032")
            with gr.Column(scale=2):
                gr.Markdown("1024x1024, 816x816, 688x688, 512x512, 416x416, 336x336")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                sintel_preview = gr.Image("./samples/Sintel/frame_0028_temple.png", label="Sintel", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("1024 x 436")
            with gr.Column(scale=2):
                gr.Markdown("1024x432, 816x352, 688x288, 512x224")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                kitti_preview = gr.Image("./samples/KITTI/2011_10_03_drive_0047_sync_image_0000000383_image_02.png", label="KITTI", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("1216 x 532")
            with gr.Column(scale=2):
                gr.Markdown("1024x352, 816x288, 688x240, 512x176")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                wild_1_preview = gr.Image("./samples/Wild/toy.jpeg", label="Wild Image 1", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("3019 x 3018")
            with gr.Column(scale=2):
                gr.Markdown("1024x1024, 816x816, 688x688, 512x512, 416x416, 336x336")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                wild_2_preview = gr.Image("./samples/Wild/hamburg.jpeg", label="Wild Image 2", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("1536 x 1920")
            with gr.Column(scale=2):
                gr.Markdown("1024x1024, 816x816, 688x688, 512x512, 416x416, 336x336")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                wild_3_preview = gr.Image("./samples/Wild/north_hill.jpeg", label="Wild Image 3", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("2320 x 2321")
            with gr.Column(scale=2):
                gr.Markdown("1024x1024, 816x816, 688x688, 512x512, 416x416, 336x336")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                wild_4_preview = gr.Image("./samples/Wild/EH.jpeg", label="Wild Image 4", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("1920 x 1080")
            with gr.Column(scale=2):
                gr.Markdown("1024x1024, 816x816, 688x688, 512x512, 416x416, 336x336")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                wild_5_preview = gr.Image("./samples/Wild/train_station.jpeg", label="Wild Image 5", height=120, interactive=False, show_label=True)
            with gr.Column(scale=1):
                gr.Markdown("1066 x 1060")
            with gr.Column(scale=2):
                gr.Markdown("1024x1024, 816x816, 688x688, 512x512, 416x416, 336x336")


    # Define click events to load images
    eth3d_preview.select(fn=lambda: Image.open("./samples/ETH3D/DSC_0243.JPG"), outputs=img_input)
    sintel_preview.select(fn=lambda: Image.open("./samples/Sintel/frame_0028_temple.png"), outputs=img_input)
    kitti_preview.select(fn=lambda: Image.open("./samples/KITTI/2011_10_03_drive_0047_sync_image_0000000383_image_02.png"), outputs=img_input)
    diode_preview.select(fn=lambda: Image.open("./samples/DIODE/00022_00195_outdoor_010_030.png"), outputs=img_input)

    wild_1_preview.select(fn=lambda: Image.open("./samples/Wild/toy.jpeg"), outputs=img_input)
    wild_2_preview.select(fn=lambda: Image.open("./samples/Wild/hamburg.jpeg"), outputs=img_input)
    wild_3_preview.select(fn=lambda: Image.open("./samples/Wild/north_hill.jpeg"), outputs=img_input)
    wild_4_preview.select(fn=lambda: Image.open("./samples/Wild/EH.jpeg"), outputs=img_input)
    wild_5_preview.select(fn=lambda: Image.open("./samples/Wild/train_station.jpeg"), outputs=img_input)


demo.launch(server_name=local_ip, server_port=6969, share=False)
