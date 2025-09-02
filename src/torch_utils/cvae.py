import torch
import torch.nn as nn

class CVAE(nn.Module):
    """
    - conditional variational autoencoder with no normalizing flows
    - basic implementation based on https://www.codegenes.net/blog/cvae-pytorch/
    """
    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        input = torch.cat([x, c], dim=1)
        h = self.encoder(input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, c):
        input = torch.cat([z, c], dim=1)
        recon_x = self.decoder(input)
        return recon_x
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar