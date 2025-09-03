import pandas as pd

def calc_patch_difference(result_patch: pd.DataFrame, actual_patch: pd.DataFrame):
    diff = {}
    for idx in result_patch.iloc[0].index:
        actual = actual_patch.iloc[0][idx]
        result = result_patch.iloc[0][idx]
        diff[idx] = result - actual
    return diff