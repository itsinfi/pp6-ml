from diva import DIVA_MAP, DIVA_ENV_TO_INDEX_MAP
import dawdreamer as daw
from utils import read_diva_value
from dawdreamer_utils.convert_parameters_description import convert_parameters_description
import pandas as pd

def change_patch(row: pd.Series, diva: daw.PluginProcessor, file: str):
    patch = []

    # convert param description for faster access
    param_desc = convert_parameters_description(diva.get_parameters_description())
    
    print('file:', file)
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    # iterate through the diva original diva patch
    for key, val in (param for param in diva.get_patch()):
        
        if key in DIVA_MAP:
            patch.append((key, read_diva_value(lines, index=key, group=DIVA_MAP[key]['group'], key=DIVA_MAP[key]['key'], param_desc=param_desc)))
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


    diva.set_patch(patch)