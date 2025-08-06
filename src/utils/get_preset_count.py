import re
import traceback

def get_preset_count(preset_files: list[str]):
    preset_count = 0
    preset_with_tags_count = 0
    preset_partly_with_tags_count = 0
    folder_specific_counts = {}

    for preset_file in preset_files:
        try:
            split_result = re.split(pattern=r"[\\/]", string=preset_file)
            folder = split_result[-2]

            with open(preset_file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()

            preset_count += 1
            
            if folder in folder_specific_counts:
                folder_specific_counts[folder]['preset_count'] += 1
            else:
                folder_specific_counts[folder] = {'preset_count': 1, 'preset_with_tags_count': 0, 'preset_partly_with_tags_count': 0}

            has_categories = False
            has_features = False
            has_character = False

            for line in lines:

                if has_categories and has_features and has_character:
                    preset_with_tags_count += 1
                    folder_specific_counts[folder]['preset_with_tags_count'] += 1
                    break

                if not has_categories and line.startswith('Categories'):
                    has_categories = True
                if not has_features and line.startswith('Features'):
                    has_features = True
                if not has_character and line.startswith('Character'):
                    has_character = True
                            
            if 0 < sum([has_character, has_categories, has_features]) < 3:
                preset_partly_with_tags_count += 1
                folder_specific_counts[folder]['preset_partly_with_tags_count'] += 1

        except Exception as e:
            print(f'error when processing {preset_file}: {e}')
            traceback.print_exc()
    
    return preset_count, preset_with_tags_count, preset_partly_with_tags_count, folder_specific_counts