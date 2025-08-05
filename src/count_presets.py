from config import DIVA_PRESET_DIR
from utils import get_all_preset_files, get_preset_count, count_duplicate_presets

# TODO: write converter from fxp to h2p
# TODO: list all counts also per folder
# TODO: add last preset from freshloops
# TODO: add preset source listing file to this repository

def main():
    preset_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR)

    preset_count, preset_with_tags_count = get_preset_count(preset_files)
    duplicates, total_duplicate_count = count_duplicate_presets(preset_files)

    print(f'Presets (total): {preset_count}\t\tPresets (w/ tags): {preset_with_tags_count}\t\tPresets (w/o tags): {preset_count - preset_with_tags_count}')

    for i, duplicate in enumerate(duplicates.items()):
        name, info = duplicate
        print(f'Duplicate {i + 1}: {info["count"]}x\t{name}\n{info["files"]}\n')
    print(f'Total duplicate presets: {total_duplicate_count}')