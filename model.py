import torch
import torch.nn as nn
from transformers import BlipForQuestionAnswering

class BlipForMultipleChoice(nn.Module):
    def __init__(self, model_name, num_choices):
        super().__init__()
        self.num_choices = num_choices
        # Load the pre-trained BLIP model for VQA
        self.blip = BlipForQuestionAnswering.from_pretrained(model_name)

    def forward(self, pixel_values, input_ids, attention_mask):
        # The input tensors have a shape like (batch_size, num_choices, seq_len)
        # We need to flatten them to (batch_size * num_choices, seq_len) to process
        batch_size = input_ids.shape[0]
        
        # Reshape inputs for the BLIP model
        pixel_values = pixel_values.view(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        # Get the outputs from the BLIP model.
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # We take the logit of the first generated token as the score for the entire sequence.
        choice_scores = outputs.logits[:, 0, :].mean(dim=1)

        # Reshape the scores back to (batch_size, num_choices)
        reshaped_scores = choice_scores.view(batch_size, -1)

        return reshaped_scores