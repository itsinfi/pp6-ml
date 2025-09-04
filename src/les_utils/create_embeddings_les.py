import dawdreamer as daw
from config import DIVA_PRESET_DIR, RENDER_DURATION, NOTE_PITCH, NOTE_VELOCITY, NOTE_DURATION
import pandas as pd
import numpy as np
from dawdreamer_utils import change_patch
import laion_clap as lc
import json
import time

def create_embeddings_les(
    row: pd.Series,
    engine: daw.RenderEngine,
    diva: daw.PluginProcessor,
    clap: lc.CLAP_Module,
):
    start_a = time.perf_counter()
    # change preset
    print(f"{DIVA_PRESET_DIR}{row['meta_location']}")

    start_b = time.perf_counter()
    change_patch(row, diva, file=f"{DIVA_PRESET_DIR}{row['meta_location']}")
    end_b = time.perf_counter()

    # load diva into engine graph
    engine.load_graph([(diva, [])])

    # config for midi
    diva.add_midi_note(note=NOTE_PITCH, velocity=NOTE_VELOCITY, start_time=0.5, duration=NOTE_DURATION) # do not set start_time to zero, there will be no audible audio signal!

    # render audio
    start_c = time.perf_counter()
    engine.render(RENDER_DURATION)
    end_c = time.perf_counter()

    # get audio, make audio signal mono, convert it to ndarray and expand its dimensionality to 2 for clap
    audio_np = np.expand_dims(np.asarray(np.mean(engine.get_audio(), axis=0), dtype=np.float32), axis=0)

    # generate audio embedding
    start_d = time.perf_counter()
    audio_embed = clap.get_audio_embedding_from_data(audio_np, use_tensor=False)
    end_d = time.perf_counter()
    row['embeddings_audio'] = json.dumps(audio_embed[0].astype(np.float32).tolist())
    end_a = time.perf_counter()
    print('a', end_a - start_a, 'b', end_b - start_b, 'c', end_c - start_c, 'd', end_d - start_d)
    return row