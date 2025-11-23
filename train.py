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
from config import (
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, WEIGHT_DECAY, OPTIMIZER,
    TRAINING_STRATEGY, MODEL_DROPOUT, LABEL_SMOOTHING, EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA, DEVICE, IS_CUDA, USE_AMP, NUM_WORKERS, PIN_MEMORY,
    PERSISTENT_WORKERS, TRAIN_SPLIT, VAL_SPLIT, OUTPUT_DIR, MODEL_SAVE_PATH,
    WANDB_PROJECT_NAME, MODEL_NAME, NUM_CHOICES, GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS, GRADIENT_CLIP_VAL, BACKBONE_LR_MULTIPLIER, CLASSIFIER_LR_MULTIPLIER
)
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
    model = BlipForMultipleChoice(
        MODEL_NAME,
        NUM_CHOICES,
        dropout_p=MODEL_DROPOUT,
    ).to(DEVICE)

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

    # Optimizer with differential learning rates for backbone vs classifier
    # Separate parameters for backbone and classifier to use different learning rates
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'blip' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': LEARNING_RATE * BACKBONE_LR_MULTIPLIER,
            'weight_decay': WEIGHT_DECAY
        })
        print(f"Backbone LR: {LEARNING_RATE * BACKBONE_LR_MULTIPLIER:.2e} ({len(backbone_params)} params)")
    
    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': LEARNING_RATE * CLASSIFIER_LR_MULTIPLIER,
            'weight_decay': WEIGHT_DECAY
        })
        print(f"Classifier LR: {LEARNING_RATE * CLASSIFIER_LR_MULTIPLIER:.2e} ({len(classifier_params)} params)")
    
    if OPTIMIZER.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=LEARNING_RATE,  # This is just for initialization, actual LRs are in param_groups
            relative_step=False,
            scale_parameter=False,
        )
        print("Using Adafactor optimizer (memory-efficient).")
    else:
        optimizer = AdamW(param_groups)
        if OPTIMIZER.lower() != "adamw":
            print(f"Unknown OPTIMIZER '{OPTIMIZER}', defaulting to AdamW.")

    # Calculate total steps accounting for gradient accumulation
    steps_per_epoch = (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = amp.GradScaler(device="cuda", enabled=USE_AMP and IS_CUDA)
    amp_device = "cuda" if IS_CUDA else "cpu"
    non_blocking = IS_CUDA and PIN_MEMORY

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_valid_batches = 0
        global_step = 0
        accumulated_grad_norms = []
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}")):
            inputs, labels = batch
            inputs = {k: v.to(DEVICE, non_blocking=non_blocking) for k, v in inputs.items()}
            labels = labels.to(DEVICE, non_blocking=non_blocking)

            with amp.autocast(device_type=amp_device, enabled=scaler.is_enabled()):
                logits = model(**inputs)
                loss = loss_fn(logits, labels)
                # Normalize loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Check for NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                print(f"WARNING: Invalid loss detected: {loss.item()}. Skipping batch.")
                continue
            
            scaler.scale(loss).backward()
            
            # Track loss (denormalized)
            batch_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                total_loss += batch_loss
                num_valid_batches += 1
            
            # Accumulate gradients
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients to prevent explosion
                scaler.unscale_(optimizer)
                
                # Calculate gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                accumulated_grad_norms.append(grad_norm.item())
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        
        # Handle remaining gradients at end of epoch if any
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            accumulated_grad_norms.append(grad_norm.item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Handle case where all batches were skipped
        if num_valid_batches == 0:
            print(f"WARNING: No valid batches in epoch {epoch+1}. Loss may be unstable.")
            avg_train_loss = float('inf')
            avg_grad_norm = 0.0
        else:
            avg_train_loss = total_loss / num_valid_batches
            avg_grad_norm = np.mean(accumulated_grad_norms) if accumulated_grad_norms else 0.0
        
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f} | Avg Gradient Norm: {avg_grad_norm:.4f}")

        # Validation
        val_accuracy = evaluate_performance(
            model,
            val_loader,
            DEVICE,
            non_blocking=non_blocking,
            amp_enabled=scaler.is_enabled(),
        )
        print(f"Epoch {epoch+1} | Validation Accuracy: {val_accuracy:.4f}")

        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss if not np.isinf(avg_train_loss) else 1e6, 
            "val_accuracy": val_accuracy,
            "gradient_norm": avg_grad_norm,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # Save the best model
        improved = val_accuracy > best_val_accuracy + EARLY_STOP_MIN_DELTA
        if improved:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        # Early stopping to prevent overfitting
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping triggered after {epochs_without_improvement} "
                f"epoch(s) without validation gain."
            )
            break

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
