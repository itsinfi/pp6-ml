import pandas as pd
import numpy as np
import json

def read_condition_data(df: pd.DataFrame):
    for col in ["embeddings_audio", "embeddings_tags"]:
        df[col] = df[col].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
    return np.concatenate([np.stack(df[col].to_numpy()) for col in [
        'embeddings_audio',
        'embeddings_tags'
    ]], axis=1)