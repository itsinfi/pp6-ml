import config
import utils

def main():
    preset_files = utils.get_all_preset_files(preset_dir=config.DIVA_PRESET_DIR)
    preset_count, preset_with_tags_count = utils.get_preset_count(preset_files)

    print(f'Presets (total): {preset_count}\tPresets (w/ tags): {preset_with_tags_count}')