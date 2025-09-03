from diva import DIVA_RESULT_TO_ENV_MAP
import numpy as np

def array_to_patch(arr: np.ndarray):
    patch = {}

    # assign all metric/numerical values
    for k, v in DIVA_RESULT_TO_ENV_MAP.items():
        patch[v] = arr[k]

    patch['env_1_model'] = max(arr[5], arr[6], arr[7]) # assign model with greatest probability for env 1 (ads, analogue or digital)
    patch['env_2_model'] = max(arr[17], arr[18], arr[19]) # assign model with greatest probability for env 2 (ads, analogue or digital)

    return patch