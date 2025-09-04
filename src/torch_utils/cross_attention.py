import torch
import torch.nn as nn
from typing import Optional

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # positional encoding for sequences
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # initialize attention in both direction
        self.audio_to_text_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout, batch_first=True)

        # projection for multimodal concatenation of the condition
        self.proj = nn.Linear(self.embed_dim * 2, self.embed_dim)

        # dummy token to handle missing inputs
        self.dummy_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def prepare_sequence(self, x: Optional[torch.Tensor]):
        # x is expected to have shape (batch_size, embed_dim)
        if x is None:
            return None
        
        # conversion to sequence with length of 1
        seq = x.unsqueeze(1) # shape here is (batch_size, seq_len=1, embed_dim)
        return seq
    
    def apply_dummy_for_missing(self, c_seq: Optional[torch.Tensor]):
        # if modality is missing -> replicate dummy token
        if c_seq is None:
            return self.dummy_token.expand(c_seq.size(0), -1, -1)
        
        mask = torch.all(c_seq == 0, dim=1, keepdim=True).bool() # shape (batch, seq_len, 1)
        mask = mask.squeeze(-1) # shape (batch, seq_len = 1)

        if mask.any():
            c_seq = c_seq.clone()
            dummy_expanded = self.dummy_token.expand(c_seq.size(0), c_seq.size(1), c_seq.size(2))
            c_seq[mask] = dummy_expanded[mask]
        return c_seq

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
        # create sequence out of audio and text inputs
        audio_seq = self.prepare_sequence(audio)
        text_seq = self.prepare_sequence(text)

        # add dummy in case their is no audio/text
        audio_seq = self.apply_dummy_for_missing(audio_seq) if audio_seq is not None else None
        text_seq = self.apply_dummy_for_missing(text_seq) if text_seq is not None else None

        if audio_seq is None and text_seq is None:
            raise ValueError('at löeast audio or text must be provided')
        
        # if only audio is provided -> use self attention for audio
        elif audio_seq is not None and text_seq is None:
            audio_out, _ = self.audio_to_text_attn(audio_seq, audio_seq, audio_seq)
            return audio_out.squeeze(1)
        
        # if only text is provided -> use self attention for text
        elif text is not None and audio is None:
            text_out, _ = self.text_to_audio_attn(text_seq, text_seq, text_seq)
            return text_out.squeeze(1)

        # use cross attention for audio and text if both were provided
        else:
            audio_out, _ = self.audio_to_text_attn(audio_seq, text_seq, text_seq)
            text_out, _ = self.text_to_audio_attn(text_seq, audio_seq, audio_seq)

            combined_out = torch.cat([audio_out, text_out], dim=-1)
            return self.proj(combined_out).squeeze(1)