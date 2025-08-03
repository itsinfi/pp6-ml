from config import DIVA_PRESET_DIR
from utils import get_all_preset_files, get_preset_count

def main():
    preset_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR)
    preset_count, preset_with_tags_count = get_preset_count(preset_files)

    print(f'Presets (total): {preset_count}\tPresets (w/ tags): {preset_with_tags_count}')