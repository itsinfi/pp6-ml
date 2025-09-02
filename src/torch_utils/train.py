import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .cvae import CVAE
from .init_weights import init_weights
from .calc_loss import calc_loss
from utils import logger
import numpy as np
from datetime import datetime

def train(x_data: np.ndarray[np.ndarray], c_data: np.ndarray[np.ndarray]):
    """
    - handles the training process for the cvae
    - basic implementation based on https://www.codegenes.net/blog/cvae-pytorch/
    """

    # convert dataset to torch tensor
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    c_tensor = torch.tensor(c_data, dtype=torch.float32)

    # calculate dataset dimensions
    input_dim = x_tensor.shape[1]
    cond_dim = c_tensor.shape[1]
    latent_dim = 128
    logger.info(f'input_dim: {input_dim}\tcond_dim: {cond_dim}\tlatent_dim: {latent_dim}')

    # apply a fixed seed to minimize randomness
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    # initialize training loader
    dataset = TensorDataset(x_tensor, c_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=False, generator=g)

    # initialize model and weights
    model = CVAE(input_dim, cond_dim, latent_dim)
    model.apply(init_weights)

    # initialize adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # configuration for training process and early stopping
    max_epochs = 500
    es_patience: int = 5
    es_min_delta: float = 1e-4

    # values to track best model and early stopping
    best_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    # training process
    for epoch in range(max_epochs):
        # initialize loss values
        train_total_loss, train_bce_loss, train_kld_loss = 0.0, 0.0, 0.0
        
        # iterate through dataset
        for x, c in train_loader:
            x = x.view(-1, input_dim)

            # core operations for training prcess
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            total_loss, bce_loss, kld_loss = calc_loss(recon_x, x, mu, logvar)
            total_loss.backward()
            optimizer.step()
            
            # track loss values
            train_total_loss += total_loss.item()
            train_bce_loss += bce_loss.item()
            train_kld_loss += kld_loss.item()

        # output loss
        num_batches = len(train_loader)
        epoch_total_loss = train_total_loss / num_batches
        logger.info(
            f"Epoch {epoch + 1}: Total Loss={epoch_total_loss:.4f}, "
            f"BCE={(train_bce_loss / num_batches):.4f}, KLD={(train_kld_loss / num_batches):.4f}"
        )

        # early stopping: track epochs with no improvement
        if best_loss - epoch_total_loss > es_min_delta:
            best_loss = epoch_total_loss
            epochs_no_improve = 0

            # save the best state of the model
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1

        # early stopping: stop if patience for epochs with no improvement is overreached
        if epochs_no_improve >= es_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # return the best model
    return {
        'model_sate_dict': best_model_state,
        'meta': {
            'datetime': datetime.now().isoformat(),
            'loss': best_loss,
            'input_dim': input_dim,
            'cond_dim': cond_dim,
            'latent_dim': latent_dim,
            'epochs': best_epoch,
        },
    }
