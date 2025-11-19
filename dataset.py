import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor
from typing import Tuple, Dict, Any

class ScienceQADataset(Dataset):
    def __init__(self, split, processor, config):
        from datasets import load_dataset # Lazy import
        self.processor = processor
        self.num_choices = config.NUM_CHOICES
        self.max_length = config.MAX_LENGTH
        self.data = load_dataset(config.DATASET_NAME, split=split)
        self.blank_image_size = self._get_blank_image_size()

    def _get_blank_image_size(self) -> Tuple[int, int]:
        """
        Determine a reasonable fallback size for placeholder images based on the
        processor configuration. BLIP processors expose the target size either
        as a dict with height/width or as a single shortest_edge value.
        """
        size = getattr(self.processor, "image_processor", None)
        size = getattr(size, "size", None)
        if isinstance(size, dict):
            height = size.get("height") or size.get("shortest_edge")
            width = size.get("width") or size.get("shortest_edge")
        else:
            height = width = size if size is not None else 224
        height = height or 224
        width = width or 224
        return int(height), int(width)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']
        image = example.get('image')

        # Create a blank image if none is provided
        if image is None:
            height, width = self.blank_image_size
            image = Image.new('RGB', (width, height), color='white')
        # Pad choices to NUM_CHOICES
        num_choices = len(choices)
        num_padding = max(0, self.num_choices - num_choices)
        if num_padding > 0:
            # Pad with empty strings (will be masked out)
            choices = choices + [''] * num_padding

        # Mask to identify which choices are valid (unpadded)
        choice_mask = [1] * num_choices + [0] * num_padding

        # The core idea: create one input sequence for each choice
        # The model will learn to score the sequence "[QUESTION] [SEP] [CHOICE]"
        text_inputs = [f"{question} [SEP] {choice}" for choice in choices]

        # Process image and text pairs
        inputs = self.processor(
            images=[image] * len(text_inputs),
            text=text_inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        inputs["choice_mask"] = torch.tensor(choice_mask, dtype=torch.bool)

        labels = torch.tensor(answer_idx, dtype=torch.long)

        return inputs, labels
