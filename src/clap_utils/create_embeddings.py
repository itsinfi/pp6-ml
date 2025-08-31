import pandas as pd
import laion_clap as lc
import numpy as np

def create_embeddings(row: pd.DataFrame, clap: lc.CLAP_Module, dataset_name: str):
    # generate audio embedding
    audio_file = f"audio/{dataset_name}/{row['meta_name']}.wav"
    audio_embed = clap.get_audio_embedding_from_filelist([audio_file], use_tensor=False)
    row['embeddings_audio'] = [emb.astype(np.float32) for emb in audio_embed]

    # read tags
    tags = row['tags_categories'] + row['tags_features'] + row['tags_character']

    # generate tag embeddings
    if tags:
        tags_embed = clap.get_text_embedding(tags, use_tensor=False)
        row['embeddings_tags'] = [emb.astype(np.float32) for emb in tags_embed]

    return row