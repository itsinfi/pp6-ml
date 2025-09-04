import pandas as pd
import laion_clap as lc
import numpy as np
import json

def create_embeddings(row: pd.Series, clap: lc.CLAP_Module, dataset_name: str):
    print('file:', row['meta_location'])

    # generate audio embedding
    audio_file = f"audio/{dataset_name}/{row['meta_name']}.wav"
    audio_embed = clap.get_audio_embedding_from_filelist([audio_file], use_tensor=False)
    row['embeddings_audio'] = json.dumps(audio_embed[0].astype(np.float32).tolist())

    # read tags
    tags = row['tags_categories'] + row['tags_features'] + row['tags_character']

    # generate tag embeddings
    if tags:
        tags_embed = clap.get_text_embedding(tags, use_tensor=False)
        row['embeddings_tags'] = json.dumps(tags_embed[0].astype(np.float32).tolist())
    else:
        row['embeddings_tags'] = json.dumps(np.zeros(512, dtype=np.float32).tolist())

    return row