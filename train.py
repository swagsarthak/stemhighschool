import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BlipProcessor, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import numpy as np

from config import *
from dataset import ScienceQADataset
from model import BlipForMultipleChoice

def train():
    """
    Main training function.
    """
    # Initialize WandB
    wandb.init(project=WANDB_PROJECT_NAME, config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "model_name": MODEL_NAME
    })

    # Load processor and model
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForMultipleChoice(MODEL_NAME, NUM_CHOICES).to(DEVICE)

    # Load datasets
    train_dataset = ScienceQADataset(split=TRAIN_SPLIT, processor=processor, config=globals())
    val_dataset = ScienceQADataset(split=VAL_SPLIT, processor=processor, config=globals())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}"):
            inputs, labels = batch
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            
            logits = model(**inputs)
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        val_accuracy = evaluate_performance(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1} | Validation Accuracy: {val_accuracy:.4f}")

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_accuracy": val_accuracy})

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

    wandb.finish()
    print("Training complete.")

def evaluate_performance(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples