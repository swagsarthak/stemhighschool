import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import BlipProcessor

import config
from config import *
from model import BlipForMultipleChoice


def _get_blank_image(processor: BlipProcessor) -> Image.Image:
    """
    Create a simple white placeholder image using the processor's expected size.
    """
    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, dict):
        height = size.get("height") or size.get("shortest_edge") or 224
        width = size.get("width") or size.get("shortest_edge") or 224
    else:
        height = width = size or 224
    return Image.new("RGB", (int(width), int(height)), color="white")


def _prepare_inputs(
    question: str,
    choices: List[str],
    processor: BlipProcessor,
    image: Image.Image,
) -> dict:
    """
    Format a single example into the tensor batch expected by the model.
    """
    if not choices:
        raise ValueError("Please provide at least one answer choice.")
    if len(choices) > config.NUM_CHOICES:
        raise ValueError(
            f"Number of choices ({len(choices)}) exceeds configured NUM_CHOICES={config.NUM_CHOICES}."
        )

    num_choices = len(choices)
    num_padding = max(0, config.NUM_CHOICES - num_choices)
    padded_choices = choices + [""] * num_padding
    choice_mask = torch.tensor([1] * num_choices + [0] * num_padding, dtype=torch.bool)

    text_inputs = [f"{question} [SEP] {choice}" for choice in padded_choices]
    inputs = processor(
        images=[image] * len(text_inputs),
        text=text_inputs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
    )
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
    inputs["choice_mask"] = choice_mask.unsqueeze(0)
    return inputs


def answer_question(
    question: str,
    choices: List[str],
    image_path: Optional[str] = None,
    model_path: str = MODEL_SAVE_PATH,
) -> Tuple[str, List[float]]:
    """
    Load the fine-tuned model and return the best choice with probabilities.
    """
    processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = BlipForMultipleChoice(MODEL_NAME, NUM_CHOICES).to(DEVICE)
    model.eval()

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. Run training first."
        )
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = _get_blank_image(processor)

    inputs = _prepare_inputs(question.strip(), choices, processor, image)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)

    probs = probs[0, : len(choices)].cpu().tolist()
    best_idx = int(torch.argmax(torch.tensor(probs)))
    return choices[best_idx], probs


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the STEM QA model.")
    parser.add_argument("--question", required=True, help="STEM question to answer.")
    parser.add_argument(
        "--choices",
        nargs="+",
        required=True,
        help="List of answer options (space separated).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional path to an accompanying image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        prediction, probabilities = answer_question(
            question=args.question,
            choices=args.choices,
            image_path=args.image_path,
        )
        print(f"Predicted answer: {prediction}")
        print("Probabilities (aligned to provided choices):")
        for choice, prob in zip(args.choices, probabilities):
            print(f"- {choice}: {prob:.3f}")
    except Exception as exc:
        print(f"Error during inference: {exc}")
