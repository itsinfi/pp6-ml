import pandas as pd
import numpy as np

def add_zero_rows(df: pd.DataFrame, key: str, valid_indices: list[int], zero_perc: float, rng: np.random.Generator):
    if zero_perc > 0 and valid_indices:
        size = int(len(valid_indices) * zero_perc)
        zero_indices = rng.choice(valid_indices, size, replace=False)
        for idx in zero_indices:
            df.at[df.index[idx], key] = np.zeros(512, dtype=np.float32)