from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_columns(df: pd.DataFrame, numeric_cols: list[str]):
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])