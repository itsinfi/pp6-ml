import sys
import pandas as pd
from torch_utils import train, save_model
from utils import logger, read_input_data, read_condition_data
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
    
    # read input data for envelope params and save them as a numpy array
    x_data = read_input_data(df)
    logger.info(f'x: {x_data[0]}')

    # read conditional data for audio and text embeddings and save them as a numpy array
    c_data = read_condition_data(df)
    logger.info(f'c: {c_data[0]}')
    
    # run training process
    model = train(x_data, c_data)

    # save model
    save_model(
        model,
        dataset_name, 
        name=f"cvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epochs-{model['meta']['epochs']}_loss-{model['meta']['loss']:.2f}"
    )