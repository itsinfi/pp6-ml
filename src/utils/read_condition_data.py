import pandas as pd
import numpy as np
import json
from .add_zero_rows import add_zero_rows

def read_condition_data(
    df: pd.DataFrame,
    audio_zero_perc: float = 0.3,
    tags_zero_perc: float = 0.0,
    seed: int = 42
):
    # parse embeddings from json
    for col in ['embeddings_audio', 'embeddings_tags']:
        df[col] = df[col].apply(lambda x: np.array(json.loads(x), dtype=np.float32))

    # copy dataframe to keep original embeddings
    df_copy = df.copy()

    # initialize random generator
    rng = np.random.default_rng(seed)

    # zero rows for audio
    valid_audio_indices = [i for i, arr in enumerate(df_copy['embeddings_tags']) if np.all(arr != 0)]
    add_zero_rows(
        df=df_copy,
        key='embeddings_audio', 
        valid_indices=valid_audio_indices, 
        zero_perc=audio_zero_perc,
        rng=rng
    )

    # zero rows for tags
    valid_tags_indices = [i for i, arr in enumerate(df_copy['embeddings_audio']) if np.all(arr != 0)]
    add_zero_rows(
        df=df_copy,
        key='embeddings_tags', 
        valid_indices=valid_tags_indices, 
        zero_perc=tags_zero_perc,
        rng=rng
    )

    # return concatenated array
    audio_embeds, text_embeds = np.stack(df_copy['embeddings_audio'].to_numpy()), np.stack(df_copy['embeddings_tags'].to_numpy())
    return audio_embeds, text_embeds