import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import librenderman as rm
import pandas as pd

def change_patch(row: pd.DataFrame, re: rm.RenderEngine, diva_preset_dir: str):
    with open(f"{diva_preset_dir}{row['meta_location']}", mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(lines)