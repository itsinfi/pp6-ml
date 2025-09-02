import pandas as pd
from typing import List
from diva import DIVA_ENV_MAX_MAP

def normalize_columns(df: pd.DataFrame, numeric_cols: List[str]):
    df_normalized = df.copy()
    for col in numeric_cols:
        max = DIVA_ENV_MAX_MAP[col]
        min = 0
        df_normalized[col] = df[col].clip(lower=min, upper=max)
        df_normalized[col] = abs(df[col] / max)
    return df_normalized