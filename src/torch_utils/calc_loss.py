import torch

def calc_loss(recon_x, x, mu, logvar):
    bce_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = bce_loss + kld_loss
    return total_loss, bce_loss, kld_loss