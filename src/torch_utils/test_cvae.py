from dawdreamer_utils import init_dawdreamer, render_patch
from clap_utils import init_clap, create_embeddings
from typing import Dict
import numpy as np
from .cvae import CVAE
import torch
from utils import logger
from diva import array_to_patch, DIVA_MAP
import time

def test_cvae(
    x_test: np.ndarray[np.ndarray[np.float32]], 
    c_test: np.ndarray[np.ndarray[np.float32]],
    model_state_dict: Dict,
    latent_dim: int
):
    # create render engine for renderman
    engine, diva = init_dawdreamer()

    # init clap
    # clap = init_clap() TODO:

    # convert datasets to torch tensor
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    c_test_tensor = torch.tensor(c_test, dtype=torch.float32)

    # calculate dataset dimensions
    input_dim = x_test_tensor.shape[1]
    cond_dim = c_test_tensor.shape[1]
    logger.info(f'input_dim: {input_dim}\tcond_dim: {cond_dim}\tlatent_dim: {latent_dim}')

    # init model
    model = CVAE(input_dim, cond_dim, latent_dim)
    model.load_state_dict(model_state_dict)
    
    # TEST PHASE --------------------------------------------------------------
    model.eval()

    timer_results = []
    patch_results = []
    clap_score_results = [] # TODO:

    with torch.no_grad():
        for x, x_tensor, c_tensor in zip(x_test, x_test_tensor, c_test_tensor):
            # convert to batch
            x_tensor = x_tensor.unsqueeze(0)
            c_tensor = c_tensor.unsqueeze(0)

            # start timer
            start = time.perf_counter()

            # generate patch
            recon, _, _ = model(x_tensor, c_tensor)

            # read generated patch data
            result_patch = array_to_patch(recon.squeeze(0).numpy())

            # stop timer
            end = time.perf_counter()
            timer_results.append(end - start)

            # read actual patch data
            actual_patch = array_to_patch(x)
            
            # calculate patch difference for each value
            diff = {}
            for idx, result in result_patch.items():
                actual = actual_patch[idx]
                diff[f"{DIVA_MAP[idx]['group']}_{DIVA_MAP[idx]['key']}"] = result - actual
            patch_results.append(diff)

            # calculate result patch embedding
            # render_patch() TODO:

            # calculate actual patch embedding


    print(f"{sum(timer_results) / len(timer_results):6f} seconds")
    print(patch_results)


