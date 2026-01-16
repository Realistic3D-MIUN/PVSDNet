import os

params_height = 256
params_width = 256
params_m = 32
params_number_input = 1
params_step_size = 5
params_gamma = 0.2

TRAIN_LOCATION = "./lf_train.txt"
VALIDATION_LOCATION = "./lf_validate.txt"
TEST_LOCATION = "./lf_test.txt"
LOG_FILE_LOCATION = "./logs/training_log_0.txt"
CHECKPOINT_LOCATION = "./checkpoint/"
RESUME_CHECKPOINT_LOCATION = "./checkpoint/checkpoint_best.pth"
START_CHECKPOINT_LOCATION = "./checkpoint/checkpoint_init.pth"
DEVICE = "cuda:0"

BATCH_SIZE = 24
LEARNING_RATE = 0.00001
NUM_EPOCHS = 50
START_EPOCH = 0
T_max = 50
PRINT_INTERVAL = 20

os.makedirs("./logs",exist_ok=True)
os.makedirs("./checkpoint",exist_ok=True)
os.makedirs("./output",exist_ok=True)


