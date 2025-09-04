from dawdreamer_utils import init_dawdreamer, render_patch
from clap_utils import init_clap, create_embeddings
import numpy as np
from utils import calc_patch_difference
from diva import array_to_patch
import time
import pandas as pd
import json
import os
from .les import les

def test_les(
    dataset_name: str,
    x_test: np.ndarray[np.ndarray[np.float32]], 
    c_test: np.ndarray[np.ndarray[np.float32]],
    df_test: pd.DataFrame,
):
    # create render engine for renderman
    engine, diva = init_dawdreamer()

    # init clap
    clap = init_clap()
    
    # TEST PHASE --------------------------------------------------------------
    results = []

    for x, c, df_test_row in zip(x_test, c_test, df_test.iterrows()):
        _, df_test_row_val = df_test_row

        result = {}

        # start timer
        start = time.perf_counter()

        # generate patch with only text input
        solution = les(c[512:], df_test_row_val, clap, engine, diva)

        # read generated patch data
        result_patch = array_to_patch(np.array(solution))

        # stop timer
        end = time.perf_counter()
        result['time'] = end - start

        # read actual patch data
        actual_patch = array_to_patch(x)
        
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

    # convert results to dataframe
    df_results = pd.json_normalize(results)

    # calculate stats
    df_stats = df_results.describe()

    # save results + stats
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{dataset_name}', exist_ok=True)
    df_results.to_parquet(f'results/{dataset_name}/les_{dataset_name}_results.parquet', compression='gzip')
    df_stats.to_csv(f'results/{dataset_name}/les_{dataset_name}_results_stats.csv')