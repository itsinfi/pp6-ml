from dawdreamer_utils import init_dawdreamer, render_patch
from clap_utils import init_clap, create_embeddings
from typing import Dict
import numpy as np
from .cvae import CVAE
import torch
from utils import logger, calc_patch_difference
from diva import array_to_patch
import time
import pandas as pd

def test_cvae(
    dataset_name: str,
    x_test: np.ndarray[np.ndarray[np.float32]], 
    c_test: np.ndarray[np.ndarray[np.float32]],
    df_test: pd.DataFrame,
    model_state_dict: Dict,
    latent_dim: int,
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
        for x, x_tensor, c_tensor, (_, df_test_row) in zip(x_test, x_test_tensor, c_test_tensor, df_test.iterrows()):

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
            patch_results.append(calc_patch_difference(result_patch, actual_patch))

            # read meta info
            result_patch['meta_name'] = df_test_row['meta_name']
            result_patch['meta_location'] = df_test_row['meta_location']

            # calculate result patch embedding
            result_dataset_name = f'{dataset_name}_results'
            render_patch(result_patch, engine, diva, dataset_name=result_dataset_name)
            # create_embeddings(result_patch, clap, dataset_name=result_dataset_name) TODO:

            # compare to actual patch embedding
            # TODO:


    print(f"{sum(timer_results) / len(timer_results):6f} seconds")
    print(patch_results)

    # TODO: convert results to dataframe, save them and their stats


