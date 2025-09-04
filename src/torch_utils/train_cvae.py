import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .cvae import CVAE
from .init_weights import init_weights
from .calc_loss import calc_loss
from utils import logger
import numpy as np
from datetime import datetime

def train_cvae(
    x_train: np.ndarray[np.ndarray[np.float32]],
    c_train: np.ndarray[np.ndarray[np.float32]],
    x_val: np.ndarray[np.ndarray[np.float32]],
    c_val: np.ndarray[np.ndarray[np.float32]]
):
    """
    - handles the training process for the cvae
    - basic implementation based on https://www.codegenes.net/blog/cvae-pytorch/
    """

    # convert datasets to torch tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    c_train_tensor = torch.tensor(c_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    c_val_tensor = torch.tensor(c_val, dtype=torch.float32)

    # calculate dataset dimensions
    input_dim = x_train_tensor.shape[1]
    cond_dim = c_train_tensor.shape[1]
    latent_dim = 64
    logger.info(f'input_dim: {input_dim}\tcond_dim: {cond_dim}\tlatent_dim: {latent_dim}')

    # apply a fixed seed to minimize randomness
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize training loader
    train_dataset = TensorDataset(x_train_tensor, c_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # initialize validation loader
    val_dataset = TensorDataset(x_val_tensor, c_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
    final_train_loss = float('inf')
    final_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    # training process
    for epoch in range(max_epochs):

        # TRAINING PHASE -------------------------------------------------------------------------------------------------------------
        model.train()

        # initialize loss values
        train_total_loss, train_bce_loss, train_kld_loss = 0.0, 0.0, 0.0
        
        # iterate through dataset
        for x_train_batch, c_train_batch in train_loader:
            x_train_batch = x_train_batch.view(-1, input_dim)

            # split audio and text
            audio_train = c_train_batch[:, :cond_dim // 2]
            text_train  = c_train_batch[:, cond_dim // 2:]

            # train batch
            optimizer.zero_grad()
            recon_train, mu_train, logvar_train = model(x_train_batch, audio_train, text_train)
            loss, bce, kld = calc_loss(recon_train, x_train_batch, mu_train, logvar_train)
            loss.backward()
            optimizer.step()
            
            # track training loss values
            train_total_loss += loss.item()
            train_bce_loss += bce.item()
            train_kld_loss += kld.item()

        # calc training loss
        num_train_batches = len(train_loader)
        train_loss_epoch = train_total_loss / num_train_batches

        # VALIDATION PHASE -----------------------------------------------------------------------------------------------------
        model.eval()

        # initialize validation loss
        val_total_loss, val_bce_loss, val_kld_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for x_val_batch, c_val_batch in val_loader:
                x_val_batch = x_val_batch.view(-1, input_dim)

                # split audio and text
                audio_val = c_val_batch[:, :cond_dim // 2]
                text_val  = c_val_batch[:, cond_dim // 2:]

                # validate batch
                recon_val, mu_val, logvar_val = model(x_val_batch, audio_val, text_val)
                loss, bce, kld = calc_loss(recon_val, x_val_batch, mu_val, logvar_val)

                # track validation loss
                val_total_loss += loss.item()
                val_bce_loss += bce.item()
                val_kld_loss += kld.item()

        # calc validation loss
        num_val_batches = len(val_loader)
        val_loss_epoch = val_total_loss / num_val_batches

        # log epoch stats
        logger.info(
            f"Epoch {epoch + 1}:\n"
            f"Train Loss->{train_loss_epoch:.4f}, "
            f"Train BCE->{(train_bce_loss / num_train_batches):.4f}, Train KLD->{(train_kld_loss / num_train_batches):.4f}\n"
            f"Validation Loss->{val_loss_epoch:.4f}, "
            f"Val BCE->{(val_bce_loss / num_train_batches):.4f}, Val KLD->{(val_kld_loss / num_train_batches):.4f}\n"
            f"{'-' * 50}"
        )

        # EARLY STOPPING -------------------------------------------------------------------------

        # track epochs with no or little validation loss improvement
        if final_val_loss - val_loss_epoch > es_min_delta:
            final_train_loss = train_total_loss
            final_val_loss = val_total_loss
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
        'model_state_dict': best_model_state,
        'meta': {
            'datetime': datetime.now().isoformat(),
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'input_dim': input_dim,
            'cond_dim': cond_dim,
            'latent_dim': latent_dim,
            'epochs': best_epoch,
        },
    }
