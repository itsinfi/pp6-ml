import pandas as pd
import numpy as np

def read_input_data(df: pd.DataFrame):
    return df[[
        'env_1_attack',
        'env_1_decay',
        'env_1_sustain',
        'env_1_release',
        'env_1_velocity',
        'env_1_model_ads',
        'env_1_model_analogue',
        'env_1_model_digital',
        'env_1_trigger',
        'env_1_quantize',
        'env_1_curve',
        'env_1_release_on',
        'env_1_key_follow',
        'env_2_attack',
        'env_2_decay',
        'env_2_sustain',
        'env_2_release',
        'env_2_velocity',
        'env_2_model_ads',
        'env_2_model_analogue',
        'env_2_model_digital',
        'env_2_trigger',
        'env_2_quantize',
        'env_2_curve',
        'env_2_release_on',
        'env_2_key_follow',
    ]].to_numpy().astype(np.float32)