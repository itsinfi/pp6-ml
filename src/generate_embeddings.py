from dawdreamer_utils import init_dawdreamer, render_patch
import pandas as pd
import sys

def main():
    """
    this script basically does two things
    - record audio for each preset via renderman
    - generate embeddings with clap out of audio and text tags if available via laion-clap

    optional argument: dataset name (use it like this: 'run_visualize_stats my_cool_dataset_name')
    """

    dataset_name = str(sys.argv[1]) if len(sys.argv) > 1 else 'dataset_raw'

    # read dataset
    try:
        df = pd.read_parquet(f'data/{dataset_name}.parquet')
    except FileNotFoundError:
        print(f'Error: data/{dataset_name}.parquet not found. make sure to run the script "read_presets" first before executing this script.')
        return

    # create render engine for renderman
    engine, diva = init_dawdreamer()

    # render patch audio
    df.apply(lambda row: render_patch(row, engine, diva, dataset_name), axis=1)

    # initialize clap model

    # generate embeddings
    # df.apply(lambda row: )