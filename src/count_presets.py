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

    total_presets, tagged_presets, semi_tagged_presets, folder_counts = get_preset_count(preset_files)
    duplicates, total_duplicates = count_duplicate_presets(preset_files)

    buffer = StringIO()
    buffer.write('Output for count_presets:\n\n\n\n\n')

    for i, folder_count in enumerate(folder_counts.items()):
        folder, counts = folder_count
        buffer.write(
            f'\n{i + 1} {folder}:\nPresets (total): {counts["total_presets"]}'
            f'\t\tPresets (w/ tags): {counts["tagged_presets"]}'
            f'\t\tPresets (w/ only some tags): {counts["semi_tagged_presets"]}'
            f'\t\tPresets (w/o tags): {counts["total_presets"] - counts["tagged_presets"] - counts["semi_tagged_presets"]}\n'
        )
        buffer.write('-' * 100)
    buffer.write(
        f'\nPresets (total): {total_presets}'
        f'\t\tPresets (w/ tags): {tagged_presets}'
        f'\t\tPresets (w/ only some tags): {semi_tagged_presets}'
        f'\t\tPresets (w/o tags): {total_presets - tagged_presets - semi_tagged_presets}\n\n\n'
    )

    for i, duplicate in enumerate(duplicates.items()):
        name, info = duplicate
        buffer.write(f'\nDuplicate {i + 1}: {info["count"]}x\t{name}\n{info["files"]}\n')
        buffer.write('-' * 100)
    buffer.write(f'\nTotal duplicate presets: {total_duplicates}\n\n\n')

    output = buffer.getvalue()

    print(output)

    file = 'data/count_presets_output.txt'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file, 'w', encoding='utf-8') as f:
        f.write(output)
