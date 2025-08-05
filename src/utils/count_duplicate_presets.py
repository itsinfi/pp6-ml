import re

def count_duplicate_presets(preset_files: list[str]):
    pattern = r"[\\/]"

    files_names = []
    duplicates = {}
    duplicate_count = 0

    for preset_file in preset_files:
        split_result = re.split(pattern, string=preset_file)
        file_name = split_result[-1]
        files_names.append(file_name)

        # in case file name is already a known duplicate (at least 2 occurences)
        if file_name in duplicates:
            duplicates[file_name] += 1
            duplicate_count += 1

        # in case file name is a first time duplicate (only 2 occurences so far)
        elif file_name in files_names[:-1]:
            duplicates[file_name] = 2
            duplicate_count += 1

    return duplicates, duplicate_count

