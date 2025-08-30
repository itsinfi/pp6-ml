from typing import List, Dict

def convert_parameters_description(param_desc: List[Dict[str, any]]):
    new_param_desc = {}

    for desc in param_desc:
        index = desc['index']
        del desc['index']
        new_param_desc[index] = desc

    return new_param_desc
