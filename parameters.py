import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 224
IMG_WIDTH = 224
SHUFFLING = True
NUM_EPOCHS = 20
BATCH_SIZE = 10
#NUM_CLASSES = 10 
GRAD_ENABLE = False
MODEL = "resnet18"
LEARNING_RATE = 8e-3
MOMENTUM = 0.9
NUM_WORKERS = 1
DATASET_HOME = "dataset/"
#LOGS_PATH = "logs"

#SAVE_MODEL = True
#LOAD_MODEL = False