import glob
import os

def get_all_preset_files(preset_dir: str):
    pathname = os.path.join(preset_dir, '**', '*.h2p')
    files = glob.glob(pathname, recursive=True)
    return files