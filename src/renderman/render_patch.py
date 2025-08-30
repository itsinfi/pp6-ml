import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import librenderman as rm
import pandas as pd
import numpy as np
import soundfile as sf
from .change_patch import change_patch

def render_patch(
    row: pd.DataFrame,
    re: rm.RenderEngine,
    dataset_name: str,
    diva_preset_dir: str,
    sample_rate: int,
    midi_note_pitch: int,
    velocity: int,
    note_length_seconds: float,
    render_length_seconds: float,
    fade_in_seconds: float,
    fade_out_seconds: float,
):
    change_patch(row, re, diva_preset_dir)

    # render patch
    re.render_patch(midi_note_pitch, velocity, note_length_seconds, render_length_seconds)

    # read rendered audio
    af = re.get_audio_frames()

    # convert to numpy array
    af_np = np.array(af, dtype=np.float32)

    # add fade in
    fade_in_samples = min(int(fade_in_seconds * sample_rate), len(af))
    fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
    af_np[-fade_in_samples:] *= fade_in_curve

    # add fade out
    fade_out_samples = min(int(fade_out_seconds * sample_rate), len(af))
    fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
    af_np[-fade_out_samples:] *= fade_out_curve
    
    # create necessary directories if neccessary
    os.makedirs('audio', exist_ok=True)
    os.makedirs(f'audio/{dataset_name}', exist_ok=True)

    # save as wav file
    sf.write(file=f"audio/{dataset_name}/{row['meta_name']}.wav", data=af_np, samplerate=sample_rate, subtype='PCM_16')