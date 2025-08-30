from diva import DIVA_MAP
import dawdreamer as daw
from utils import read_diva_value
from dawdreamer_utils.convert_parameters_description import convert_parameters_description

def change_patch(diva: daw.PluginProcessor, file: str):
    patch = []

    # convert param description for faster access
    param_desc = convert_parameters_description(diva.get_parameters_description())

    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    for key, val in (param for param in diva.get_patch()):
        
        #TODO:
        # if key in DIVA_MAP:
        #     patch.append((key, read_diva_value(lines, index=key, group=DIVA_MAP[key]['group'], key=DIVA_MAP[key]['key'], param_desc=param_desc)))
        #     continue
        
        patch.append((key, val))

    # diva.set_patch(patch)
    diva.set_patch(diva.get_patch())