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
        self.dummy_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.dummy_token)

    def prepare_sequence(self, x: Optional[torch.Tensor]):
        # x is expected to have shape (batch_size, embed_dim)
        if x is None:
            return None
        
        # conversion to sequence with length of 1
        seq = x.unsqueeze(1) # shape here is (batch_size, seq_len=1, embed_dim)
        return seq
    
    def apply_dummy_for_missing(self, c_seq: Optional[torch.Tensor], batch_size: int):
        # if modality is missing -> replicate dummy token
        if c_seq is None:
            dummy = self.dummy_token.expand(batch_size, 1, self.embed_dim).clone()
            dummy += torch.randn_like(dummy) * 1e-6
            return dummy
        
        mask = torch.all(c_seq == 0, dim=-1, keepdim=True) # shape (batch, seq_len, 1)
        mask = mask.squeeze(-1) # shape (batch, seq_len = 1)

        if mask.any():
            c_seq = c_seq.clone()
            dummy_expanded = self.dummy_token.expand_as(c_seq).clone()
            dummy_expanded += torch.randn_like(dummy_expanded) * 1e-6
            c_seq[mask] = dummy_expanded[mask]

        # correct possible case of all values being 0 and therefore all values are now nan due to softmax
        c_seq = torch.nan_to_num(c_seq, nan=0.0, posinf=1e6, neginf=-1e6)
        return c_seq

    def get_mask(self, tensor: Optional[torch.TensorType]):
        if tensor is None:
            return None
        
        # calculate if all elements are zero
        mask = torch.all(tensor == 0, dim=1, keepdim=True).bool()
        return mask
    
    def get_batch_size(self, audio: Optional[torch.Tensor], text: Optional[torch.Tensor]):
        if audio is not None:
            return audio.size(0)
        if text is not None:
            return text.size(0)
        raise ValueError("audio or text must be provided")

    def forward(
        self,
        audio: Optional[torch.TensorType] = None,
        text: Optional[torch.TensorType] = None,
    ):
        batch_size = self.get_batch_size(audio, text)
        
        # create sequence out of audio and text inputs
        audio_seq = self.prepare_sequence(audio)
        text_seq = self.prepare_sequence(text)

        print('1_aud', audio_seq, '1_txt', text_seq)

        # add dummy in case their is no audio/text
        audio_seq = self.apply_dummy_for_missing(audio_seq, batch_size)
        text_seq = self.apply_dummy_for_missing(text_seq, batch_size)
        print('2_aud', audio_seq, '2_txt', text_seq)

        if audio_seq is None and text_seq is None:
            raise ValueError('at löeast audio or text must be provided')
        
        # if only audio is provided -> use self attention for audio
        elif audio_seq is not None and text_seq is None:
            audio_out, _ = self.audio_to_text_attn(audio_seq, audio_seq, audio_seq)
            print('3_aud', audio_out)
            return audio_out.squeeze(1)
        
        # if only text is provided -> use self attention for text
        elif text is not None and audio is None:
            text_out, _ = self.text_to_audio_attn(text_seq, text_seq, text_seq)
            print('4_txt', text_out)
            return text_out.squeeze(1)

        # use cross attention for audio and text if both were provided
        else:
            audio_out, _ = self.audio_to_text_attn(audio_seq, text_seq, text_seq)
            text_out, _ = self.text_to_audio_attn(text_seq, audio_seq, audio_seq)

            print('5_aud', audio_out, '5_txt', text_out)

            combined_out = torch.cat([audio_out, text_out], dim=-1)
            print('6', combined_out)
            return self.proj(combined_out).squeeze(1)