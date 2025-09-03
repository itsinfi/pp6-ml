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
import json
import os

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
    clap = init_clap()

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

    results = []

    with torch.no_grad():
        for x, c, df_test_row in zip(x_test_tensor, c_test_tensor, df_test.iterrows()):
            _, df_test_row_val = df_test_row

            result = {}

            # convert to batch
            x = x.unsqueeze(0)
            c = c.unsqueeze(0)

            # start timer
            start = time.perf_counter()

            # generate patch with only text input
            recon, _, _ = model(x, c) # TODO: add cross attention and fix this

            # read generated patch data
            result_patch = array_to_patch(recon.squeeze(0).numpy())

            # stop timer
            end = time.perf_counter()
            result['time'] = end - start

            # read actual patch data
            actual_patch = array_to_patch(x.squeeze(0).numpy())
            
            # calculate patch difference for each value
            result['patch_similarity'] = calc_patch_difference(result_patch, actual_patch)

            # read meta info
            result_patch['meta_name'] = df_test_row_val['meta_name']
            result_patch['meta_location'] = df_test_row_val['meta_location']
            result_patch['tags_categories'] = df_test_row_val['tags_categories']
            result_patch['tags_features'] = df_test_row_val['tags_features']
            result_patch['tags_character'] = df_test_row_val['tags_character']

            # convert result patch to dataframe
            df_result_patch = pd.Series(result_patch)

            # calculate result patch embedding
            result_dataset_name = f'{dataset_name}_results'
            render_patch(df_result_patch, engine, diva, dataset_name=result_dataset_name)
            df_result_patch = create_embeddings(df_result_patch, clap, dataset_name=result_dataset_name)

            # compare to actual patch embedding
            result_embed = np.array(json.loads(df_result_patch['embeddings_audio']), dtype=np.float32)
            actual_embed = df_test_row_val['embeddings_audio']
            cos_sim = np.dot(result_embed, actual_embed) / (np.linalg.norm(result_embed) * np.linalg.norm(actual_embed))
            result['text_clap_score'] = cos_sim

            # append result
            results.append(result)

    print(results)

    # convert results to dataframe
    df_results = pd.json_normalize(results)

    # calculate stats
    df_stats = df_results.describe()

    # save results + stats
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{dataset_name}', exist_ok=True)
    df_results.to_parquet(f'results/cvae_{dataset_name}_results.parquet', compression='gzip')
    df_stats.to_csv(f'results/cvae_{dataset_name}_results_stats.csv')


