import dawdreamer as daw
from config import DIVA_PRESET_DIR, SAMPLE_RATE, RENDER_DURATION, FADE_IN_DURATION, FADE_OUT_DURATION, NOTE_PITCH, NOTE_VELOCITY, NOTE_DURATION
import os
import pandas as pd
import numpy as np
import soundfile as sf
from .change_patch import change_patch

def render_patch(row: pd.DataFrame, engine: daw.RenderEngine, diva: daw.PluginProcessor, dataset_name: str):
    # change preset
    # print(diva.get_patch()[:285])
    change_patch(diva, file=f"{DIVA_PRESET_DIR}{row['meta_location']}")
    # print(diva.get_patch()[:285])

    # config for midi
    diva.add_midi_note(note=NOTE_PITCH, velocity=NOTE_VELOCITY, start_time=0., duration=NOTE_DURATION)

    # load diva into engine graph
    engine.load_graph([(diva, [])])

    # render audio
    engine.render(RENDER_DURATION)

    # get audio
    audio = engine.get_audio()

    # convert to numpy array and make it mono
    audio_np = np.mean(audio, axis=0).astype(np.float32)
    print('max:', audio_np.max())

    # TODO: add fade in
    # fade_in_samples = min(int(FADE_IN_DURATION * SAMPLE_RATE), len(audio))
    # fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
    # audio_np[-fade_in_samples:] *= fade_in_curve

    # TODO: add fade out
    # fade_out_samples = min(int(FADE_OUT_DURATION * SAMPLE_RATE), len(audio))
    # fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
    # audio_np[-fade_out_samples:] *= fade_out_curve
    
    # create necessary directories if neccessary
    os.makedirs('audio', exist_ok=True)
    os.makedirs(f'audio/{dataset_name}', exist_ok=True)

    # save as wav file
    sf.write(file=f"audio/{dataset_name}/{row['meta_name']}.wav", data=audio_np, samplerate=SAMPLE_RATE, subtype='PCM_16')