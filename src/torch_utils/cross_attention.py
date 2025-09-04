import torch
import torch.nn as nn
from typing import Optional

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()

        # initializes multihead attention for supporting multimodality
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # projection for multimodal concatenation of the condition
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def get_mask(self, tensor: Optional[torch.TensorType]):
        if tensor is None:
            return None
        
        # calculate if all elements are zero
        mask = torch.all(tensor == 0, dim=1, keepdim=True).bool()
        return mask

    def forward(
        self,
        audio: Optional[torch.TensorType] = None,
        text: Optional[torch.TensorType] = None
    ):
        audio_mask = self.get_mask(audio)
        text_mask = self.get_mask(text)

        if audio is not None:
            audio = audio.unsqueeze(1)
        if text is not None:
            text = text.unsqueeze(1)

        if audio is None and text is None:
            raise ValueError('at löeast audio or text must be provided')
        
        # use only audio in case no text was provided (self attention)
        elif audio is not None and text is None:
            audio_out, _ = self.attn(audio, audio, audio, key_padding_mask=audio_mask)
            audio_out = torch.nan_to_num(audio_out, nan=0.0)

            return audio_out.squeeze(1)
        # only use text in case no text was provided (self attention)
        elif text is not None and audio is None:
            text_out, _ = self.attn(text, text, text, key_padding_mask=text_mask)
            text_out = torch.nan_to_num(text_out, nan=0.0)

            return text_out.squeeze(1)

        # use cross attention for audio and text if both were provided and concatenate output
        else:
            audio_out, _ = self.attn(audio, text, text, key_padding_mask=text_mask)
            audio_out = torch.nan_to_num(audio_out, nan=0.0)

            text_out, _ = self.attn(text, audio, audio, key_padding_mask=audio_mask)
            text_out = torch.nan_to_num(text_out, nan=0.0)

            combined_out = torch.cat([audio_out.squeeze(1), text_out.squeeze(1)], dim=-1)
            return self.proj(combined_out)