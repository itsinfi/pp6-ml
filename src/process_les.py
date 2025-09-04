import sys
import pandas as pd
from les_utils import test_les
from utils import read_input_data, read_condition_data, split_data

def main():
    """
    handles testing the les model

    optional argument: dataset name (use it like this: 'run_train_cvae my_cool_dataset_name')
    """

    dataset_name = str(sys.argv[1]) if len(sys.argv) > 1 else 'dataset'

    # read dataset
    try:
        df = pd.read_parquet(f'data/{dataset_name}_embedded.parquet')
    except FileNotFoundError:
        print(f'Error: data/{dataset_name}_embedded.parquet not found. make sure to run the script "generate_embeddings" first before executing this script.')
        return
    
    # split data
    _, _, df_test = split_data(df)
    
    # read input data for envelope params and save them as a numpy array
    x_test = read_input_data(df_test)

    # read conditional data for audio and text embeddings and save them as a numpy array
    audio_test, text_test = read_condition_data(df_test, audio_zero_perc=0.0)

    # run les algorithm
    test_les(dataset_name, x_test, audio_test, text_test, df_test)