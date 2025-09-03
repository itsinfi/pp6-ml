from typing import Dict

def calc_patch_difference(result_patch: Dict, actual_patch: Dict):
    diff = {}
    for idx, result in result_patch.items():
        diff[idx] = result - actual_patch[idx]
    return diff