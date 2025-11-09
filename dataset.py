import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor

class ScienceQADataset(Dataset):
    def __init__(self, split, processor, config):
        from datasets import load_dataset # Lazy import
        self.processor = processor
        self.config = config
        self.data = load_dataset(config.DATASET_NAME, split=split)

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
            image = Image.new('RGB', (self.processor.image_processor.size['height'], self.processor.image_processor.size['width']), color='white')
        # Pad choices to NUM_CHOICES
        num_choices = len(choices)
        num_padding = self.config.NUM_CHOICES - num_choices
        if num_padding > 0:
            # Pad with empty strings (will be masked out)
            choices = choices + [''] * num_padding

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
            max_length=self.config.MAX_LENGTH
        )

        labels = torch.tensor(answer_idx, dtype=torch.long)

        return inputs, labels