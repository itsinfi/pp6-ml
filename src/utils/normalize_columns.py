from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import List

def normalize_columns(df: pd.DataFrame, numeric_cols: List[str]):
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])