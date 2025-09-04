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

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ):
        if audio is None and text is None:
            raise ValueError('at löeast audio or text must be provided')
        
        # use only audio in case no text was provided (self attention)
        elif audio is not None and text is None:
            audio = audio.unsqueeze(1)
            audio_out, _ = self.attn(audio, audio, audio)
            return audio_out.squeeze(1)
        
        # only use text in case no text was provided (self attention)
        elif text is not None and audio is None:
            text = text.unsqueeze(1)
            text_out, _ = self.attn(text, text, text)
            return text_out.squeeze(1)

        # concatenate audio and text if both were provided (cross attention)
        else:
            audio = audio.unsqueeze(1)
            text = text.unsqueeze(1)
            audio_out, _ = self.attn(audio, text, text)
            text_out, _ = self.attn(text, audio, audio)
            return self.proj(torch.cat([
                audio_out.squeeze(1),
                text_out.squeeze(1)
            ], dim=-1))