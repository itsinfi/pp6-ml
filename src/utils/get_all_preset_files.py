import glob
import os

def get_all_preset_files(preset_dir: str, file_type: str = 'h2p'):
    pathname = os.path.join(preset_dir, '**', f'*.{file_type}')
    files = glob.glob(pathname, recursive=True)
    return files