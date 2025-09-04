import torch
import torch.nn as nn
from .cross_attention import CrossAttention
from typing import Optional

class CVAE(nn.Module):
    """
    - conditional variational autoencoder with cross attention and no normalizing flows
    - basic implementation based on https://www.codegenes.net/blog/cvae-pytorch/
    """
    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int):
        super(CVAE, self).__init__()

        # dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim // 2
        
        # initialize cross attention mechanism
        self.cross_attn = CrossAttention(embed_dim=self.cond_dim)
    
        # encoder for extracting most important features out of the input and conditions
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.cond_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # for creating a mean vector for latent space
        self.fc_mu = nn.Linear(128, latent_dim)

        # for creting a log variation vector for latent space
        self.fc_logvar = nn.Linear(128, latent_dim)

        # decoder for reconstructing the patch from a simplified patch and conditions to understand how to build it
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.cond_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.input_dim),
            nn.Sigmoid(),
        )

    def encode(
        self,
        x: Optional[torch.TensorType],
        audio: Optional[torch.TensorType],
        text: Optional[torch.TensorType]
    ):
        # apply cross attention
        c = self.cross_attn(audio, text)

        # concatenate input and condition
        input = torch.cat([x, c], dim=1)

        # calculate h (latent space)
        h = self.encoder(input)

        # calculate base for mean vector for h
        mu = self.fc_mu(h)

        # calculate base for log variation vector for h
        logvar = self.fc_logvar(h)

        return mu, logvar, c
    
    def reparameterize(self, mu: nn.Linear, logvar: nn.Linear):
        # calculate standard deviation
        std = torch.exp(0.5 * logvar)

        # generate random noise sampled from std
        eps = torch.randn_like(std)

        # calculate latent sample
        return mu + eps * std
    
    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))
    
    def forward(
        self,
        x: Optional[torch.TensorType] = None,
        audio: Optional[torch.TensorType] = None,
        text: Optional[torch.TensorType] = None
    ):
        # generate a patch from a provided starting point
        if x is not None:
            mu, logvar, c = self.encode(x, audio, text)
            z = self.reparameterize(mu, logvar)

        # generate a patch from scratch
        else:
            c = self.cross_attn(audio, text)
            batch_size = c.size(0)
            device = c.device
            z = torch.randn(batch_size, self.fc_mu.out_features, device=device)
            mu, logvar = None, None
        
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar