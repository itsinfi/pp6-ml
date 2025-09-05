import dawdreamer as daw
from config import RENDER_DURATION, NOTE_PITCH, NOTE_VELOCITY, NOTE_DURATION
import pandas as pd
import numpy as np
from dawdreamer_utils import read_diva_value
import laion_clap as lc
from diva_utils import DIVA_MAP, DIVA_ENV_TO_INDEX_MAP
from typing import Dict, Any

def create_embeddings_les(
    row: pd.DataFrame,
    preset: list[str],
    engine: daw.RenderEngine,
    diva: daw.PluginProcessor,
    clap: lc.CLAP_Module,
    param_desc: Dict[int, Dict[str, Any]]
):  
    patch = []

    # iterate through the diva original diva patch
    for key, val in (param for param in diva.get_patch()):
        
        if key in DIVA_MAP:
            patch.append((key, read_diva_value(lines=preset, index=key, group=DIVA_MAP[key]['group'], key=DIVA_MAP[key]['key'], param_desc=param_desc)))
            continue
        
        patch.append((key, val))

    # iterate through the envelope params from the dataframe and replace params in patch
    for key, val in row.items():

        if key in DIVA_ENV_TO_INDEX_MAP:
            idx = DIVA_ENV_TO_INDEX_MAP[key]

            for i, (k, _) in enumerate(patch):
                if k == idx:
                    patch[i] = (k, val)
                    break

    # change preset
    diva.set_patch(patch)

    # load diva into engine graph
    engine.load_graph([(diva, [])])

    # config for midi
    diva.add_midi_note(
        note=NOTE_PITCH, velocity=NOTE_VELOCITY, start_time=0.5, duration=NOTE_DURATION) # do not set start_time to zero, there will be no audible audio signal!

    # render audio
    engine.render(RENDER_DURATION)

    # get audio, make audio signal mono, convert it to ndarray and expand its dimensionality to 2 for clap
    audio_np = np.expand_dims(np.asarray(np.mean(engine.get_audio(), axis=0), dtype=np.float32), axis=0)

    # generate audio embedding
    audio_embed = clap.get_audio_embedding_from_data(audio_np, use_tensor=False)
    return np.asarray(audio_embed[0], dtype=np.float32)