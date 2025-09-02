import sys
import pandas as pd
from torch_utils import train, save_model
import numpy as np
from utils import logger
from datetime import datetime

def main():
    """
    handles training the cvae model

    optional argument: dataset name (use it like this: 'run_train_cvae my_cool_dataset_name')
    """

    dataset_name = str(sys.argv[1]) if len(sys.argv) > 1 else 'dataset_raw'

    # read dataset
    try:
        df = pd.read_parquet(f'data/{dataset_name}_embedded.parquet')
    except FileNotFoundError:
        print(f'Error: data/{dataset_name}_embedded.parquet not found. make sure to run the script "generate_embeddings" first before executing this script.')
        return
    
    # filter input data for envelope params and save them as a numpy array
    x_data = df[[
        'env_1_attack',
        'env_1_decay',
        'env_1_sustain',
        'env_1_release',
        'env_1_velocity',
        'env_1_model_ads',
        'env_1_model_analogue',
        'env_1_model_digital',
        'env_1_trigger',
        'env_1_quantize',
        'env_1_curve',
        'env_1_release_on',
        'env_1_key_follow',
        'env_2_attack',
        'env_2_decay',
        'env_2_sustain',
        'env_2_release',
        'env_2_velocity',
        'env_2_model_ads',
        'env_2_model_analogue',
        'env_2_model_digital',
        'env_2_trigger',
        'env_2_quantize',
        'env_2_curve',
        'env_2_release_on',
        'env_2_key_follow',
    ]].to_numpy().astype(np.float32)

    # filter conditional data for audio and text embeddings and save them as a numpy array
    c_data = np.concatenate([np.stack(df[col].to_numpy()) for col in [
        'embeddings_audio',
        'embeddings_tags'
    ]], axis=1)

    logger.info(f'x: {x_data[0]}')
    logger.info(f'c: {c_data[0]}')
    
    # run training process
    model = train(x_data, c_data)

    # save model
    save_model(model, dataset_name, name=f"cvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}")