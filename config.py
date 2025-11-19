import torch

# -- Training & Model Parameters --
MODEL_NAME = "Salesforce/blip-vqa-base"
NUM_CHOICES = 5  # Max number of choices in ScienceQA
LEARNING_RATE = 3e-5
BATCH_SIZE = 2  # Lower to reduce GPU/CPU memory pressure
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw"  # Options: "adamw", "adafactor"
TRAINING_STRATEGY = "full"  # Options: "full", "linear_probe" (freeze BLIP and train classifier only)
FORCE_CUDA = True  # Set to True to fail fast if CUDA is unavailable
if FORCE_CUDA:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Install a CUDA-enabled PyTorch build.")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"
MAX_LENGTH = 128 # Max token length for text
USE_AMP = IS_CUDA  # Enable automatic mixed precision only when on GPU
NUM_WORKERS = 0  # Use 0 on Windows/low RAM to avoid DataLoader worker overhead
PIN_MEMORY = False  # Disable to reduce memory spikes on CPU-heavy setups
PERSISTENT_WORKERS = False

# -- Dataset Parameters --
DATASET_NAME = "derek-thomas/ScienceQA"
# Use full splits for training/eval
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

# -- Paths --
OUTPUT_DIR = "outputs/saved_models/"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/scienceqa-blip-mc-best.pt"

# -- WandB Logging --
WANDB_PROJECT_NAME = "ScienceQA-Graduate-Project"
