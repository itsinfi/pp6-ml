import re

def count_duplicate_presets(preset_files: list[str]):
    pattern = r"[\\/]"

    file_names = []
    duplicates = {}
    total_duplicate_count = 0

    for preset_file in preset_files:
        split_result = re.split(pattern, string=preset_file)
        file_name = split_result[-1]
        file_names.append(file_name)

        # in case file name is already a known duplicate (at least 2 occurences)
        if file_name in duplicates:
            index = file_names.__len__() - 1

            duplicates[file_name]['files'].append(preset_files[index])
            duplicates[file_name]['count'] += 1

            total_duplicate_count += 1

        # in case file name is a first time duplicate (only 2 occurences so far)
        elif file_name in file_names[:-1]:
            index_1 = file_names[:-1].index(file_name)
            index_2 = file_names.__len__() - 1

            duplicates[file_name] = {
                'count': 2, 
                'files': [
                    preset_files[index_1], 
                    preset_files[index_2]
                ]
            }

            total_duplicate_count += 1

    return duplicates, total_duplicate_count

