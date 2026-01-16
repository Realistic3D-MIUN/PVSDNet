import os
import argparse
import numpy as np
import torch
from models.pvsdnet_model import PVSDNet
from models.pvsdnet_lite_model import PVSDNet_Lite
import helperFunctions as helper
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import parameters_pvsdnet as params

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_type', type=str, default='pvsdnet', choices=['pvsdnet', 'pvsdnet_lite'], help="Choose the model type between 'pvsdnet' and 'pvsdnet_lite'")
parser.add_argument('--checkpoint', type=str, default="./checkpoint/pvsdnet_model.pth", help="Checkpoint path for pvsdnet")
parser.add_argument('--checkpoint_lite', type=str, default="./checkpoint/pvsdnet_lite_model.pth", help="Checkpoint path for pvsdnet_lite")
parser.add_argument('--input_image', type=str, default="./samples/PVSDNet_Samples/person.jpeg", help="Location of input image")
parser.add_argument('--output_location', type=str, default="./output/Video/", help="Location of output video")
parser.add_argument('--output_file', type=str, default="video.mp4", help="Name of output video")
parser.add_argument('--radius', type=float, default=0.06, help="Radius of the circular path")
parser.add_argument('--num_frames', type=int, default=120, help="Number of frames per loop")
parser.add_argument('--num_loops', type=int, default=3, help="Total loops")

opt, _ = parser.parse_known_args()

INPUT_IMAGE_LOCATION = opt.input_image
OUTPUT_LOCATION = opt.output_location
OUTPUT_FILE = opt.output_file

os.makedirs(OUTPUT_LOCATION, exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def getPositionVector(x, y, z):
    """
    Normalize x, y, z from [-0.1, 0.1] into [0, 1] and return as a vector.
    """
    vector = torch.zeros((1, 3), dtype=torch.float)
    normalized_x = (float(format(x, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_y = (float(format(y, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    normalized_z = (float(format(z, '.7f')) - (-0.1)) / (0.1 - (-0.1))
    vector[0][0] = normalized_x
    vector[0][1] = normalized_y
    vector[0][2] = normalized_z
    return vector

def stack_images_side_by_side(im1, im2, im3):
    """
    Given three PIL images (im1: synthesized view, im2: raw depth, im3: colored depth),
    create and return a new image with all three stacked horizontally.
    """
    # Ensure all images have the same height
    if im1.height != im2.height or im1.height != im3.height:
        raise ValueError("All images must have the same height to stack them side by side.")
    
    combined_width = im1.width + im2.width + im3.width
    combined_image = Image.new("RGB", (combined_width, im1.height))
    combined_image.paste(im1, (0, 0))
    combined_image.paste(im2, (im1.width, 0))
    combined_image.paste(im3, (im1.width + im2.width, 0))
    return combined_image

def predictAndStackImage(model, input_location, target_pose):
    """
    Load the input image, synthesize the novel view at the given target pose, apply depth estimation,
    convert the depth to both raw and colored versions, and stack the three images side by side.
    """
    # Transform input image
    transform = transforms.Compose([
        transforms.Resize((params.params_height, params.params_width)),
        transforms.ToTensor()
    ])
    img_input = Image.open(input_location).convert('RGB')
    img_input = transform(img_input)

    # Get the target pose (x, y, z) and prepare position vector
    x, y, z = target_pose
    output_position = getPositionVector(x, y, z).unsqueeze(0).to(params.DEVICE)
    img_input = img_input.unsqueeze(0).to(params.DEVICE)

    with torch.no_grad():
        # Generate synthesized view
        pred_img, depth_img = model(img_input, output_position)
        depth_img = (depth_img-depth_img.min())/(depth_img.max()-depth_img.min())
        

    # Detach and move to CPU
    pred_img = pred_img.detach().cpu()
    depth_img = depth_img.detach().cpu()

    # Convert synthesized image and raw depth to PIL images
    im_pred = transforms.ToPILImage()(pred_img[0])
    im_depth_raw = transforms.ToPILImage()(depth_img[0])
    
    # Create a colored depth map using the 'magma' colormap
    depth_np = depth_img.squeeze().numpy()
    # Normalize depth (avoid division by zero)
    if np.max(depth_np) > 0:
        depth_norm = depth_np / np.max(depth_np)
    else:
        depth_norm = depth_np
    depth_colored = plt.get_cmap('inferno')(depth_norm)[:, :, :3]  # remove alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)
    im_depth_colored = Image.fromarray(depth_colored)
    
    # Stack the three images side by side
    combined = stack_images_side_by_side(im_pred, im_depth_raw, im_depth_colored)
    return combined

def generateCircularTrajectory(radius, num_frames, num_loops):
    """
    Generate a list of [x, y, z] positions representing a circular trajectory.
    The camera moves in a circle around the origin. The path repeats `num_loops` times.
    """
    angles = np.linspace(0, 2 * np.pi * num_loops, num_frames * num_loops)
    trajectory = []
    for angle in angles:
        x = radius * np.cos(angle)
        y = 0.0  # Fixed height
        z = radius * np.sin(angle)
        trajectory.append([x, y, z])
    return trajectory

def create_video(output_dir, output_file, fps=60):
    """
    Create a video file from images stored in the output directory.
    """
    images = [img for img in sorted(os.listdir(output_dir)) if img.endswith(".jpg")]
    if len(images) == 0:
        raise ValueError("No images found in output directory to create video.")
    
    frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(output_dir, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

# -----------------------------
# Load Models
# -----------------------------
# Load the view synthesis model
if opt.model_type == 'pvsdnet_lite':
    model = PVSDNet_Lite(total_image_input=params.params_number_input)
    model = helper.load_Checkpoint(opt.checkpoint_lite, model, load_cpu=True)
else:
    model = PVSDNet(total_image_input=params.params_number_input)
    model = helper.load_Checkpoint(opt.checkpoint, model, load_cpu=True)
model.to(params.DEVICE)
model.eval()

# -----------------------------
# Parameters for Circular Motion
# -----------------------------
radius = opt.radius       # Adjust for a bigger/smaller circular path
num_frames = opt.num_frames     # Frames per loop
num_loops = opt.num_loops       # Total loops

# Generate the circular trajectory (each pose is [x, y, z])
# Generate the circular trajectory for ONE loop only
final_trajectory = generateCircularTrajectory(radius, num_frames, 1)

# -----------------------------
# Generate and Save Frames
# -----------------------------
print("Generating frames with synthesized view and depth maps...")
one_loop_images = []
total_frames = num_frames * num_loops

# Initialize progress bar
pbar = tqdm(total=total_frames, desc="Generating Frames")

for loop in range(num_loops):
    for i, pose in enumerate(final_trajectory):
        # Calculate global frame index
        frame_idx = loop * num_frames + i
        
        if loop == 0:
            # First loop: Infer and save
            combined_image = predictAndStackImage(model, INPUT_IMAGE_LOCATION, pose)
            one_loop_images.append(combined_image)
        else:
            # Subsequent loops: Reuse
            combined_image = one_loop_images[i]
        
        # Save to disk (required for create_video)
        combined_image.save(f"{OUTPUT_LOCATION}{frame_idx:06d}.jpg")
        pbar.update(1)

pbar.close()

# -----------------------------
# Create Video from Frames
# -----------------------------
video_output_file = os.path.join(OUTPUT_LOCATION, OUTPUT_FILE)
create_video(OUTPUT_LOCATION, video_output_file)
print(f"Video saved to {video_output_file}")
