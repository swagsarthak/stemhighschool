import torch
import torch.nn as nn
from transformers import BlipForQuestionAnswering

class BlipForMultipleChoice(nn.Module):
    def __init__(self, model_name, num_choices, dropout_p=0.1):
        super().__init__()
        self.num_choices = num_choices
        # Load the pre-trained BLIP model for VQA
        self.blip = BlipForQuestionAnswering.from_pretrained(model_name)
        hidden_size = self.blip.config.text_config.hidden_size
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values, input_ids, attention_mask, choice_mask=None):
        # The input tensors have a shape like (batch_size, num_choices, seq_len)
        # We need to flatten them to (batch_size * num_choices, seq_len) to process
        batch_size = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        
        # Reshape inputs for the BLIP model
        pixel_values = pixel_values.view(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        # Encode the image first, then condition the text encoder on image embeddings via cross-attention
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=True)
        encoder_hidden_states = vision_outputs.last_hidden_state
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        text_outputs = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        # CLS token (or first token) carries a summary representation of the multimodal pair
        pooled_output = text_outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Reshape the scores back to (batch_size, num_choices)
        reshaped_scores = logits.view(batch_size, num_choices)

        # Mask out padded choices so they do not affect loss or predictions
        if choice_mask is not None:
            choice_mask = choice_mask.view(batch_size, num_choices)
            reshaped_scores = reshaped_scores.masked_fill(choice_mask == 0, float("-inf"))

        return reshaped_scores
