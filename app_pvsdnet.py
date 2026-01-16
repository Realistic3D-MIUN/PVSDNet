import gradio as gr
import torch
import numpy as np
import cv2
import tempfile
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.pvsdnet_model import PVSDNet
from models.pvsdnet_lite_model import PVSDNet_Lite
import helperFunctions as helper
import parameters_pvsdnet as params

MODEL_PVSDNET_LOCATION = "./checkpoint/pvsdnet_model.pth"
MODEL_PVSDNET_LITE_LOCATION = "./checkpoint/pvsdnet_lite_model.pth"

DEVICE = params.DEVICE

def getPositionVector(x, y, z):
    vector = torch.zeros((1, 3), dtype=torch.float)
    normalized_x = (float(format(x, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_y = (float(format(y, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_z = (float(format(z, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    vector[0][0] = normalized_x
    vector[0][1] = normalized_y
    vector[0][2] = normalized_z
    return vector

def generateCircularTrajectory(radius, num_frames):
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    return [[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles]

def generateSwingTrajectory(radius, num_frames):
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    return [[radius * np.cos(angle), 0, radius * np.sin(angle)] for angle in angles]

def create_video_from_memory(frames, fps=30):
    if not frames:
        return None
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return temp_video.name

def process_image(img, video_type, radius, num_frames, num_loops, model_type):
    if img is None:
        return None, None
        
    height, width = 256, 256

    min_dim = min(img.width, img.height)
    left = (img.width - min_dim) / 2
    top = (img.height - min_dim) / 2
    right = (img.width + min_dim) / 2
    bottom = (img.height + min_dim) / 2
    img = img.crop((left, top, right, bottom))

    if model_type == "PVSDNet Lite":
        model = PVSDNet_Lite(total_image_input=params.params_number_input)
        checkpoint = MODEL_PVSDNET_LITE_LOCATION
    else:
        model = PVSDNet(total_image_input=params.params_number_input)
        checkpoint = MODEL_PVSDNET_LOCATION
    
    try:
        model = helper.load_Checkpoint(checkpoint, model, load_cpu=True)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint}: {e}")
        
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    
    img_input = img.convert('RGB')
    img_input = transform(img_input).unsqueeze(0).to(DEVICE)

    if video_type == "Circle":
        raw_traj = generateCircularTrajectory(radius, num_frames)
        trajectory = [(p[0], p[1], 0) for p in raw_traj]
            
    elif video_type == "Swing":
        raw_traj = generateSwingTrajectory(radius, num_frames)
        trajectory = raw_traj
    else:
         raw_traj = generateCircularTrajectory(radius, num_frames)
         trajectory = [(p[0], p[1], 0) for p in raw_traj]

    view_frames = []
    depth_frames = []

# Run inference for a single loop (trajectory) to save computation
    for x, y, z in trajectory:
        pos = getPositionVector(x, y, z).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predicted_img, predicted_depth = model(img_input, pos)
        
        p_img = predicted_img[0].detach().cpu().permute(1, 2, 0).numpy()
        p_img = np.clip(p_img, 0, 1)
        p_img = (p_img * 255).astype(np.uint8)
        p_img_bgr = cv2.cvtColor(p_img, cv2.COLOR_RGB2BGR)
        view_frames.append(p_img_bgr)

        d_img = predicted_depth.squeeze().detach().cpu().numpy()
        d_min, d_max = d_img.min(), d_img.max()
        if d_max - d_min > 1e-6:
             d_img = (d_img - d_min) / (d_max - d_min)
        else:
             d_img = np.zeros_like(d_img)
        
        d_img_colored = plt.get_cmap('inferno')(d_img)[:, :, :3]
        d_img_colored = (d_img_colored * 255).astype(np.uint8)
        d_img_bgr = cv2.cvtColor(d_img_colored, cv2.COLOR_RGB2BGR)
        depth_frames.append(d_img_bgr)

    # Repeat the frames for the requested number of loops
    view_frames = view_frames * int(num_loops)
    depth_frames = depth_frames * int(num_loops)

    fps = 60
    view_video_path = create_video_from_memory(view_frames, fps=fps)
    depth_video_path = create_video_from_memory(depth_frames, fps=fps)

    return view_video_path, depth_video_path

with gr.Blocks(title="PVSDNet: View & Depth Synthesis", theme="default") as demo:
    gr.Markdown(
    """
    ## PVSDNet: Joint Depth Prediction and View Synthesis via Shared Latent Spaces in Real-Time
    * Upload an image and get a mini video showing capability of novel view and depth synthesis.
    
    **Note:** Huggingface demo is running on CPU so inference speeds will be slow. Inference might take around 2 mins.
    ### Head to our [Project Page](https://realistic3d-miun.github.io/PVSDNet/) for more details about the models.
    """)
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Input Image", height=256)
            
            with gr.Group():
                video_type = gr.Dropdown(["Circle", "Swing"], label="Trajectory Type", value="Swing")
                model_type = gr.Dropdown(["PVSDNet", "PVSDNet Lite"], label="Model Type", value="PVSDNet")
            
            with gr.Accordion("Advanced Settings", open=False):
                radius = gr.Slider(0.01, 0.1, value=0.06, label="Motion Radius")
                num_frames = gr.Slider(10, 120, value=60, step=1, label="Frames per Loop")
                num_loops = gr.Slider(1, 5, value=2, step=1, label="Number of Loops")
            
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Generated View Video", height=256)
            depth_video_output = gr.Video(label="Generated Depth Video", height=256)

    submit_btn.click(
        fn=process_image,
        inputs=[img_input, video_type, radius, num_frames, num_loops, model_type],
        outputs=[video_output, depth_video_output]
    )

    gr.Markdown("### Example Images: Click to Load")
    with gr.Column():
        with gr.Row():
            sample_1 = gr.Image("./samples/PVSDNet_Samples/COCO_59_source_image.png", label="COCO Example 59", height=150, interactive=False, show_label=True)
            sample_2 = gr.Image("./samples/PVSDNet_Samples/COCO_16_source_image.png", label="COCO Example 16", height=150, interactive=False, show_label=True)
            sample_3 = gr.Image("./samples/PVSDNet_Samples/COCO_755_source_image.png", label="COCO Example 755", height=150, interactive=False, show_label=True)
        
        with gr.Row():
            sample_4 = gr.Image("./samples/PVSDNet_Samples/COCO_223_source_image.png", label="COCO Example 223", height=150, interactive=False, show_label=True)
            sample_5 = gr.Image("./samples/PVSDNet_Samples/COCO_23_source_image.png", label="COCO Example 23", height=150, interactive=False, show_label=True)
            sample_6 = gr.Image("./samples/PVSDNet_Samples/person.jpeg", label="Person", height=150, interactive=False, show_label=True)
            

        with gr.Row():
            sample_7 = gr.Image("./samples/PVSDNet_Samples/flower.png", label="Flower", height=150, interactive=False, show_label=True)
            sample_8 = gr.Image("./samples/PVSDNet_Samples/person_2.jpeg", label="Person", height=150, interactive=False, show_label=True)
            sample_9 = gr.Image("./samples/PVSDNet_Samples/bakery.jpeg", label="Bakery", height=150, interactive=False, show_label=True)
            
    sample_1.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/COCO_59_source_image.png"), outputs=img_input)
    sample_2.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/COCO_16_source_image.png"), outputs=img_input)
    sample_3.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/COCO_755_source_image.png"), outputs=img_input)

    sample_4.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/COCO_223_source_image.png"), outputs=img_input)
    sample_5.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/COCO_23_source_image.png"), outputs=img_input) 
    sample_6.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/person.jpeg"), outputs=img_input)

    sample_7.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/flower.png"), outputs=img_input)
    sample_8.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/person_2.jpeg"), outputs=img_input)
    sample_9.select(fn=lambda: Image.open("./samples/PVSDNet_Samples/bakery.jpeg"), outputs=img_input)
    
if __name__ == "__main__":
    demo.launch(share=True)






