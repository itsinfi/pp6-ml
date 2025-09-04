from dawdreamer_utils import init_dawdreamer, render_patch
from clap_utils import init_clap, create_embeddings
from typing import Dict
import numpy as np
from .cvae import CVAE
import torch
from utils import logger, calc_patch_difference
from diva_utils import array_to_patch
import time
import pandas as pd
import json
import os

def test_cvae(
    dataset_name: str,
    x_test: np.ndarray[np.ndarray[np.float32]], 
    audio_test: np.ndarray[np.ndarray[np.float32]],
    text_test: np.ndarray[np.ndarray[np.float32]],
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
    audio_test_tensor = torch.tensor(audio_test, dtype=torch.float32)
    text_test_tensor = torch.tensor(text_test, dtype=torch.float32)

    # calculate dataset dimensions
    input_dim = x_test_tensor.shape[1]
    cond_dim = text_test_tensor.shape[1] + audio_test_tensor.shape[1]
    logger.info(f'input_dim: {input_dim}\tcond_dim: {cond_dim}\tlatent_dim: {latent_dim}')

    # init model
    model = CVAE(input_dim, cond_dim, latent_dim)
    model.load_state_dict(model_state_dict)
    
    # TEST PHASE --------------------------------------------------------------
    model.eval()

    results = []

    with torch.no_grad():
        for x, audio_test_element, text_test_element, df_test_row in zip(x_test_tensor, audio_test_tensor, text_test_tensor, df_test.iterrows()):
            _, df_test_row_val = df_test_row

            result = {}

            # convert input to batch
            x = x.unsqueeze(0)

            # convert conditions to batch
            text = text_test_element.unsqueeze(0)
            audio = audio_test_element.unsqueeze(0)

            # skip text based iteration if text is empty
            if not torch.all(text == 0):

                # start timer
                start = time.perf_counter()

                # generate patch with only text input
                recon, _, _ = model(text=text)

                # read generated patch data
                result_patch = array_to_patch(recon.squeeze(0).numpy())

                # stop timer
                end = time.perf_counter()
                result['text_time'] = end - start

                # read actual patch data
                actual_patch = array_to_patch(x.squeeze(0).numpy())
                
                # calculate patch difference for each value
                result['text_patch_similarity'] = calc_patch_difference(result_patch, actual_patch)

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

            # skip audio based iteration if audio is empty (should not happen though)
            if not torch.all(audio == 0):

                # start timer
                start = time.perf_counter()

                # generate patch with only audio input
                recon, _, _ = model(audio=audio)

                # read generated patch data
                result_patch = array_to_patch(recon.squeeze(0).numpy())

                # stop timer
                end = time.perf_counter()
                result['audio_time'] = end - start

                # read actual patch data
                actual_patch = array_to_patch(x.squeeze(0).numpy())
                
                # calculate patch difference for each value
                result['audio_patch_similarity'] = calc_patch_difference(result_patch, actual_patch)

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
                result['audio_clap_score'] = cos_sim

            # append result
            results.append(result)

    # convert results to dataframe
    df_results = pd.json_normalize(results)

    # calculate stats
    df_stats = df_results.describe()

    # save results + stats
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{dataset_name}', exist_ok=True)
    df_results.to_parquet(f'results/{dataset_name}/cvae_{dataset_name}_results.parquet', compression='gzip')
    df_stats.to_csv(f'results/{dataset_name}/cvae_{dataset_name}_results_stats.csv')