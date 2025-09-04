import torch

def calc_loss(recon_x, x, mu, logvar):
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = mse_loss + kld_loss
    return total_loss, mse_loss, kld_loss