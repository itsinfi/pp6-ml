from config import DIVA_PRESET_DIR
from utils import get_all_preset_files, get_preset_count, count_duplicate_presets

def main():
    preset_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR)

    preset_count, preset_with_tags_count = get_preset_count(preset_files)
    duplicates, duplicate_count = count_duplicate_presets(preset_files)

    print(f'Presets (total): {preset_count}\t\tPresets (w/ tags): {preset_with_tags_count}\t\tPresets (w/o tags): {preset_count - preset_with_tags_count}')

    for duplicate, count in duplicates.items():
        print(f'Duplicate: {count}x\t{duplicate}')
    print(f'Total duplicate presets: {duplicate_count}')