import torch

# -- Training & Model Parameters --
MODEL_NAME = "Salesforce/blip-vqa-base"
NUM_CHOICES = 5  # Max number of choices in ScienceQA
LEARNING_RATE = 5e-6  # Much lower to prevent exploding gradients and overfitting
BACKBONE_LR_MULTIPLIER = 0.1  # Use 10% of main LR for pre-trained backbone (fine-tuning)
CLASSIFIER_LR_MULTIPLIER = 1.0  # Use full LR for new classifier head
BATCH_SIZE = 4  # Increased from 2 for more stable gradients
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 4  # Limited epochs to prevent overfitting while allowing sufficient training
WEIGHT_DECAY = 0.05  # Increased weight decay for stronger regularization
OPTIMIZER = "adamw"  # Options: "adamw", "adafactor"
TRAINING_STRATEGY = "full"  # Options: "full", "linear_probe" (freeze BLIP and train classifier only)
MODEL_DROPOUT = 0.3  # Increased dropout for stronger regularization
LABEL_SMOOTHING = 0.1  # Increased label smoothing for better generalization
WARMUP_STEPS = 100  # Learning rate warmup steps for stability
GRADIENT_CLIP_VAL = 1.0  # Clip gradients to prevent explosion
EARLY_STOP_PATIENCE = 5  # Increased patience to allow more training
EARLY_STOP_MIN_DELTA = 0.001  # Minimum improvement threshold (0.1%)
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
