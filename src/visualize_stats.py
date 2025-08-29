import pandas as pd
import matplotlib.pyplot as plt
import math
import sys 

def main():
    """
    visualizes the stats for the dataset stored in 'data/dataset.parquet', throws an error if not existing
    
    optional argument: dataset name (use it like this: 'run_visualize_stats my_cool_dataset_name')
    """

    dataset_name = str(sys.argv[1]) if len(sys.argv) > 1 else 'dataset_raw'

    # read dataset
    try:
        df = pd.read_parquet(f'data/{dataset_name}.parquet')
    except FileNotFoundError:
        print(f'Error: data/{dataset_name}.parquet not found. make sure to run the script "read_presets" first before executing this script.')
        return

    # seperate numeric and not numeric columns
    non_numeric_cols = [c for c in df.columns if c.startswith('meta_') or c.startswith('tags_')]
    numeric_cols = [c for c in df.columns if c not in non_numeric_cols]

    # calculate rows and columns
    col_count = 5
    row_count = math.ceil(len(numeric_cols) / col_count)

    # plot definition
    fig, axes = plt.subplots(row_count, col_count, figsize=(col_count*5, row_count*8))
    axes = axes.flatten()
    
    # change label sizes and rotations
    for i, col in enumerate(numeric_cols):
        df.boxplot(column=col, ax=axes[i])
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)
        
    # hide empty subplots (last row)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # plot and render as png
    plt.tight_layout()
    plt.savefig('data/dataset_boxplot.png', dpi=150)
    plt.close()