import sys
import pandas as pd
from torch_utils import train_cvae, save_model, test_cvae
from utils import read_input_data, read_condition_data, split_data
from datetime import datetime

def main():
    """
    handles training, validating and testing the cvae model

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
    df_train, df_val, df_test = split_data(df)
    
    # read input data for envelope params and save them as a numpy array
    x_train = read_input_data(df_train)
    x_val = read_input_data(df_val)
    x_test = read_input_data(df_test)

    # read conditional data for audio and text embeddings and save them as a numpy array
    c_train = read_condition_data(df_train)
    c_val = read_condition_data(df_val)
    c_test = read_condition_data(df_test)
    
    # run training (incl. validation)
    model = train_cvae(x_train, c_train, x_val, c_val)

    # run test
    test_cvae(
        dataset_name, 
        x_test, 
        c_test, 
        df_test,
        model_state_dict=model['model_state_dict'], 
        latent_dim=model['meta']['latent_dim']
    )

    # save model
    save_model(
        model,
        dataset_name, 
        name=f"cvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epochs-{model['meta']['epochs']}"
        f"_t-loss-{model['meta']['train_loss']:.2f}_v-loss-{model['meta']['val_loss']:.2f}"
    )