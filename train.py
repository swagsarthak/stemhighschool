import os
import sys
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Adafactor, BlipProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Ensure the local WandB run directory (./wandb) does not shadow the installed package
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path = [
    p for p in sys.path
    if Path(p or ".").resolve() != PROJECT_ROOT
] + [str(PROJECT_ROOT)]
import wandb

import config
from config import *
from dataset import ScienceQADataset
from model import BlipForMultipleChoice

def train():
    """
    Main training function.
    """
    if IS_CUDA:
        torch.backends.cudnn.benchmark = True

    # Initialize WandB (offline mode to avoid interactive prompts)
    wandb.init(project=WANDB_PROJECT_NAME, mode="offline", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "model_name": MODEL_NAME
    })

    # Ensure the output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load processor and model
    processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = BlipForMultipleChoice(MODEL_NAME, NUM_CHOICES).to(DEVICE)

    # Apply optional lighter-weight strategy (skip fine-tuning BLIP weights)
    training_strategy = TRAINING_STRATEGY.lower()
    if training_strategy == "linear_probe":
        for _, param in model.blip.named_parameters():
            param.requires_grad = False
        model.blip.eval()  # Disable dropout/other training-only layers in the frozen backbone
        print("Using linear-probe strategy: BLIP backbone frozen, training classifier head only.")
    elif training_strategy != "full":
        raise ValueError(f"Unknown TRAINING_STRATEGY '{TRAINING_STRATEGY}'. Use 'full' or 'linear_probe'.")

    # Load datasets
    train_dataset = ScienceQADataset(split=TRAIN_SPLIT, processor=processor, config=config)
    val_dataset = ScienceQADataset(split=VAL_SPLIT, processor=processor, config=config)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    # Optimizer and Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER.lower() == "adafactor":
        optimizer = Adafactor(
            trainable_params,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            relative_step=False,  # Keep explicit LR for reproducibility
            scale_parameter=False,
        )
        print("Using Adafactor optimizer (memory-efficient).")
    else:
        optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        if OPTIMIZER.lower() != "adamw":
            print(f"Unknown OPTIMIZER '{OPTIMIZER}', defaulting to AdamW.")

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    scaler = amp.GradScaler(device="cuda", enabled=USE_AMP and IS_CUDA)
    amp_device = "cuda" if IS_CUDA else "cpu"
    non_blocking = IS_CUDA and PIN_MEMORY

    best_val_accuracy = 0.0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}"):
            inputs, labels = batch
            inputs = {k: v.to(DEVICE, non_blocking=non_blocking) for k, v in inputs.items()}
            labels = labels.to(DEVICE, non_blocking=non_blocking)

            optimizer.zero_grad()

            with amp.autocast(device_type=amp_device, enabled=scaler.is_enabled()):
                logits = model(**inputs)
                loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        val_accuracy = evaluate_performance(
            model,
            val_loader,
            DEVICE,
            non_blocking=non_blocking,
            amp_enabled=scaler.is_enabled(),
        )
        print(f"Epoch {epoch+1} | Validation Accuracy: {val_accuracy:.4f}")

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_accuracy": val_accuracy})

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

    wandb.finish()
    print("Training complete.")

def evaluate_performance(model, data_loader, device, non_blocking=False, amp_enabled=False):
    model.eval()
    total_correct = 0
    total_samples = 0
    device_type = getattr(device, "type", str(device))
    amp_device = "cuda" if device_type.startswith("cuda") else "cpu"
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            inputs, labels = batch
            inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in inputs.items()}
            labels = labels.to(device, non_blocking=non_blocking)
            with amp.autocast(device_type=amp_device, enabled=amp_enabled):
                logits = model(**inputs)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples


if __name__ == "__main__":
    # Entry point for running training from the CLI
    train()
