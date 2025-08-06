import re
import traceback

def get_preset_count(preset_files: list[str]):
    total_presets = 0

    tagged_presets = 0
    semi_tagged_presets = 0

    category_tagged_presets = 0
    feature_tagged_presets = 0
    character_tagged_presets = 0

    folder_counts = {}

    for preset_file in preset_files:
        try:
            split_result = re.split(pattern=r"[\\/]", string=preset_file)
            folder = split_result[-2]

            with open(preset_file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()

            total_presets += 1
            
            if folder in folder_counts:
                folder_counts[folder]['total_presets'] += 1
            
            else:
                folder_counts[folder] = {
                    'total_presets': 1,
                    'tagged_presets': 0,
                    'semi_tagged_presets': 0,
                    'category_tagged_presets': 0,
                    'feature_tagged_presets': 0,
                    'character_tagged_presets': 0,
                }

            has_categories = False
            has_features = False
            has_character = False

            for line in lines:

                if not has_categories and line.startswith('Categories'):
                    has_categories = True
                    category_tagged_presets += 1
                    folder_counts[folder]['category_tagged_presets'] += 1
                
                if not has_features and line.startswith('Features'):
                    has_features = True
                    feature_tagged_presets += 1
                    folder_counts[folder]['feature_tagged_presets'] += 1
                
                if not has_character and line.startswith('Character'):
                    has_character = True
                    character_tagged_presets += 1
                    folder_counts[folder]['character_tagged_presets'] += 1

                if has_categories and has_features and has_character:
                    tagged_presets += 1
                    folder_counts[folder]['tagged_presets'] += 1
                    break
                            
            if 0 < sum([has_character, has_categories, has_features]) < 3:
                semi_tagged_presets += 1
                folder_counts[folder]['semi_tagged_presets'] += 1

        except Exception as e:
            print(f'error when processing {preset_file}: {e}')
            traceback.print_exc()
    
    return total_presets, tagged_presets, semi_tagged_presets, category_tagged_presets, feature_tagged_presets, character_tagged_presets, folder_counts