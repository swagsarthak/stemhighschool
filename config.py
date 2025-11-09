import torch

# -- Training & Model Parameters --
MODEL_NAME = "Salesforce/blip-vqa-base"
NUM_CHOICES = 5  # Max number of choices in ScienceQA
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128 # Max token length for text

# -- Dataset Parameters --
DATASET_NAME = "derek-thomas/ScienceQA"
# Use smaller splits for quick testing, None for full dataset
TRAIN_SPLIT = "train[:5%]" # e.g., "train" for full
VAL_SPLIT = "validation[:20%]" # e.g., "validation" for full
TEST_SPLIT = "test[:20%]" # e.g., "test" for full

# -- Paths --
OUTPUT_DIR = "outputs/saved_models/"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/scienceqa-blip-mc-best.pt"

# -- WandB Logging --
WANDB_PROJECT_NAME = "ScienceQA-Graduate-Project"