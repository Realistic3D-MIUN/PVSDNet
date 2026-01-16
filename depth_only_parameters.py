import os

params_height = 384
params_width = 384

params_number_input = 1

LOG_FILE_LOCATION = "./logs/training_log_0.txt"
CHECKPOINT_LOCATION = "./checkpoint/"
DEVICE = "cuda:0"
ONNX_PATH = "./checkpoint_onnx"

MODEL_Large_Location = "./checkpoint/depth_only_model.pth"
MODEL_Small_Location = "./checkpoint/depth_only_lite_model.pth"

os.makedirs(ONNX_PATH,exist_ok=True)
os.makedirs("./logs",exist_ok=True)
os.makedirs("./checkpoint",exist_ok=True)
os.makedirs("./output",exist_ok=True)


