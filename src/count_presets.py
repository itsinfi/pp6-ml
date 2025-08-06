from config import DIVA_PRESET_DIR
from utils import get_all_preset_files, get_preset_count, count_duplicate_presets
import os
from io import StringIO

# TODO: fix preset folder structure
# TODO: nks support

def main():
    """
    calculates various counts for presets, presets with and without category and duplicates and saves it as data/count_presets_output.txt
    """

    preset_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR)

    preset_count, preset_with_tags_count, preset_partly_with_tags_count, folder_specific_counts = get_preset_count(preset_files)
    duplicates, total_duplicate_count = count_duplicate_presets(preset_files)

    buffer = StringIO()
    buffer.write('Output for count_presets:\n\n\n\n\n')

    for i, folder_specific_count in enumerate(folder_specific_counts.items()):
        folder, stats = folder_specific_count
        buffer.write(
            f'\n{i + 1} {folder}:\nPresets (total): {stats["preset_count"]}'
            f'\t\tPresets (w/ tags): {stats["preset_with_tags_count"]}'
            f'\t\tPresets (w/ only some tags): {stats["preset_partly_with_tags_count"]}'
            f'\t\tPresets (w/o tags): {stats["preset_count"] - stats["preset_with_tags_count"] - stats["preset_partly_with_tags_count"]}\n'
        )
        buffer.write('-' * 100)
    buffer.write(
        f'\nPresets (total): {preset_count}'
        f'\t\tPresets (w/ tags): {preset_with_tags_count}'
        f'\t\tPresets (w/ only some tags): {preset_partly_with_tags_count}'
        f'\t\tPresets (w/o tags): {preset_count - preset_with_tags_count - preset_partly_with_tags_count}\n\n\n'
    )

    for i, duplicate in enumerate(duplicates.items()):
        name, info = duplicate
        buffer.write(f'\nDuplicate {i + 1}: {info["count"]}x\t{name}\n{info["files"]}\n')
        buffer.write('-' * 100)
    buffer.write(f'\nTotal duplicate presets: {total_duplicate_count}\n\n\n')

    output = buffer.getvalue()

    print(output)

    file = 'data/count_presets_output.txt'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file, 'w', encoding='utf-8') as f:
        f.write(output)
