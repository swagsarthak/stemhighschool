import torch
import torch.nn as nn
from transformers import BlipForQuestionAnswering

class BlipForMultipleChoice(nn.Module):
    def __init__(self, model_name, num_choices):
        super().__init__()
        self.num_choices = num_choices
        # Load the pre-trained BLIP model for VQA
        self.blip = BlipForQuestionAnswering.from_pretrained(model_name)
        # Get the decoder's start token id for creating decoder_input_ids
        # BLIP decoder typically uses pad_token_id as the start token
        # Try to get it from config, otherwise use a default
        if hasattr(self.blip.config, 'decoder_start_token_id') and self.blip.config.decoder_start_token_id is not None:
            self.decoder_start_token_id = self.blip.config.decoder_start_token_id
        elif hasattr(self.blip.config, 'pad_token_id') and self.blip.config.pad_token_id is not None:
            self.decoder_start_token_id = self.blip.config.pad_token_id
        else:
            # Default for BLIP models (BERT tokenizer pad token)
            self.decoder_start_token_id = 0

    def forward(self, pixel_values, input_ids, attention_mask):
        # The input tensors have a shape like (batch_size, num_choices, seq_len)
        # We need to flatten them to (batch_size * num_choices, seq_len) to process
        batch_size = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        
        # Reshape inputs for the BLIP model
        pixel_values = pixel_values.view(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        # Create decoder_input_ids (start token for each sequence)
        # For inference, we just need the start token
        batch_size_flat = pixel_values.shape[0]
        decoder_input_ids = torch.full(
            (batch_size_flat, 1),
            fill_value=self.decoder_start_token_id,
            dtype=torch.long,
            device=pixel_values.device
        )

        # Get the outputs from the BLIP model.
        # For BlipForQuestionAnswering, we need to pass labels or use generate
        # Instead, let's use the encoder outputs to get representations
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        
        # Extract the last hidden state from the decoder
        # The output should have last_hidden_state or we use the text encoder output
        if hasattr(outputs, 'logits'):
            # If logits exist, use them
            choice_scores = outputs.logits[:, 0, :].mean(dim=1)
        elif hasattr(outputs, 'last_hidden_state'):
            # Use the mean of last hidden state as score
            choice_scores = outputs.last_hidden_state.mean(dim=1).mean(dim=1)
        else:
            # Fallback: use the text encoder's output
            # Access the underlying text model
            text_outputs = self.blip.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Use the mean of the last hidden state
            choice_scores = text_outputs.last_hidden_state.mean(dim=1)

        # Reshape the scores back to (batch_size, num_choices)
        reshaped_scores = choice_scores.view(batch_size, num_choices)

        return reshaped_scores