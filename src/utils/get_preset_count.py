def get_preset_count(preset_files: list[str]):
    preset_count = 0
    preset_with_tags_count = 0

    for preset_file in preset_files:
        try:
            with open(preset_file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()

                preset_count += 1

                has_categories = False
                has_features = False
                has_character = False

                for line in lines:

                    if has_categories and has_features and has_character:
                        preset_with_tags_count += 1
                        break

                    if not has_categories and line.startswith('Categories'):
                        has_categories = True
                    if not has_features and line.startswith('Features'):
                        has_features = True
                    if not has_character and line.startswith('Character'):
                        has_character = True
                            
        except Exception as e:
            print(f'error when processing {preset_file}: {e}')
    
    return preset_count, preset_with_tags_count