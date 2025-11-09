import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor
from tqdm import tqdm
from sklearn.metrics import classification_report

import config
from config import *
from dataset import ScienceQADataset
from model import BlipForMultipleChoice

def run_evaluation():
    """
    Loads the best saved model and evaluates it on the test set.
    """
    print("--- Starting Evaluation on Test Set ---")
    
    # 1. Load processor and model
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForMultipleChoice(MODEL_NAME, NUM_CHOICES).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Successfully loaded model from {MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}.")
        print("Please run the training script first to generate a model file.")
        return

    # 2. Load the test dataset
    test_dataset = ScienceQADataset(split=TEST_SPLIT, processor=processor, config=config)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. Run evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = batch
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            logits = model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            predictions = torch.argmax(logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate and print metrics
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    print(f"\nTest Set Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    # The choices are 0-indexed, so we can create target names like 'Choice 0', 'Choice 1', etc.
    target_names = [f'Choice {i}' for i in range(NUM_CHOICES)]
    # Use `zero_division=0` to avoid warnings if a class is never predicted
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)
    print("--- Evaluation Complete ---")