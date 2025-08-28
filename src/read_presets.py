from config import DIVA_PRESET_DIR
from utils import get_all_preset_files, read_meta_tag_value, read_numerical_envelope_value, read_categorical_envelope_value, normalize_columns, remove_duplicates
import re
import pandas as pd

def main():
    """
    - reads all preset files
    - filters relevant values (categories, features, character + params for both envs)
    - normalizes numerical values (min = 0 and max = 1) and one-hot-encodes categorical values
    - saves presets incl file location as parquet files
    """

    # regex pattern for preset name
    name_re = r"([^\\/:*?\"<>|\r\n]+)(?=\.[^.]+$)"

    # regex patterns for meta data
    meta_re = re.compile(r"/\*\s*@Meta(.*?)\*/", re.DOTALL | re.IGNORECASE)
    tags_categories_re = re.compile(r"(?mi)^\s*Categories\s*:\s*'([^']*)'")
    tags_features_re = re.compile(r"(?mi)^\s*Features\s*:\s*'([^']*)'")
    tags_character_re = re.compile(r"(?mi)^\s*Character\s*:\s*'([^']*)'")

    # regex patterns for envelope params
    env_re = re.compile(r"(?mi)^#cm=ENV([12])\b(.*?)(?=^\s*#cm=|\Z)", re.DOTALL | re.IGNORECASE)
    env_attack_re = re.compile(r"(?mi)^\s*Atk\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_decay_re = re.compile(r"(?mi)^\s*Dec\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_sustain_re = re.compile(r"(?mi)^\s*Sus\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_release_re = re.compile(r"(?mi)^\s*Rel\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_velocity_re = re.compile(r"(?mi)^\s*Vel\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_model_re = re.compile(r"(?mi)^\s*Model\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_trigger_re = re.compile(r"(?mi)^\s*Trig\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_quantize_re = re.compile(r"(?mi)^\s*Quant\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_curve_re = re.compile(r"(?mi)^\s*Crve\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_release_on_re = re.compile(r"(?mi)^\s*RelOn\s*=\s*([+-]?\d+(?:\.\d+)?)")
    env_key_follow_re = re.compile(r"(?mi)^\s*KeyFlw\s*=\s*([+-]?\d+(?:\.\d+)?)")
    
    # read all presets
    preset_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR)

    patches = []

    for preset_file in preset_files:

        # init dataset params for patch
        patch = {
            'meta_name': '', # patch name
            'meta_location': '',
            'tags_categories': '', # type of tag
            'tags_features': '', # type of tag
            'tags_character': '', # type of tag
            'env_1_attack': 0,
            'env_1_decay': 0,
            'env_1_sustain': 0,
            'env_1_release': 0, # only on digital and analogue models
            'env_1_velocity': 0,
            'env_1_model_ads': 0, # model = 0
            'env_1_model_analogue': 0, # model = 1
            'env_1_model_digital': 0, # model = 2
            'env_1_trigger': 0,
            'env_1_quantize': 0, # only for digital model
            'env_1_curve': 0, # only for digital model
            'env_1_release_on': 0, # only for ads model
            'env_1_key_follow': 0,
            'env_2_attack': 0,
            'env_2_decay': 0,
            'env_2_sustain': 0,
            'env_2_release': 0, # only on digital and analogue models
            'env_2_velocity': 0,
            'env_2_model_ads': 0, # model = 0
            'env_2_model_analogue': 0, # model = 1
            'env_2_model_digital': 0, # model = 2
            'env_2_trigger': 0,
            'env_2_quantize': 0, # only for digital model
            'env_2_curve': 0, # only for digital model
            'env_2_release_on': 0, # only for ads model
            'env_2_key_follow': 0,
        }

        # read preset file
        with open(preset_file, mode='r', encoding='utf-8') as f:
            txt = f.read()

        # read file location
        patch['meta_location'] = (
            preset_file.lower()[len(DIVA_PRESET_DIR):]
            if preset_file.lower().startswith(DIVA_PRESET_DIR.lower())
            else preset_file.lower()
        )

        # read preset name
        name_match = re.search(name_re, preset_file)
        if name_match:
            patch['meta_name'] = name_match.group(0)

        # read tags from meta data
        meta_match = meta_re.search(txt)
        if meta_match:
            meta_data = meta_match.group(1)

            patch['tags_categories'] = read_meta_tag_value(meta_data, re=tags_categories_re)
            patch['tags_features'] = read_meta_tag_value(meta_data, re=tags_features_re)
            patch['tags_character'] = read_meta_tag_value(meta_data, re=tags_character_re)

        # read enveleope model from envelope data with custom one hot encoder
        env_1_model, env_2_model = read_categorical_envelope_value(txt, env_re, val_re=env_model_re, encoder={'0': 0.0, '1': 0.0, '2': 0.0})
        patch['env_1_model_ads'] = env_1_model['0']
        patch['env_1_model_analogue'] = env_1_model['1']
        patch['env_1_model_digital'] = env_1_model['2']
        patch['env_2_model_ads'] = env_2_model['0']
        patch['env_2_model_analogue'] = env_2_model['1']
        patch['env_2_model_digital'] = env_2_model['2']
        
        # read numeric params from envelope data
        patch['env_1_attack'], patch['env_2_attack'] = read_numerical_envelope_value(txt, env_re, val_re=env_attack_re)
        patch['env_1_decay'], patch['env_2_decay'] = read_numerical_envelope_value(txt, env_re, val_re=env_decay_re)
        patch['env_1_sustain'], patch['env_2_sustain'] = read_numerical_envelope_value(txt, env_re, val_re=env_sustain_re)
        patch['env_1_release'], patch['env_2_release'] = read_numerical_envelope_value(txt, env_re, val_re=env_release_re)
        patch['env_1_velocity'], patch['env_2_velocity'] = read_numerical_envelope_value(txt, env_re, val_re=env_velocity_re)
        patch['env_1_trigger'], patch['env_2_trigger'] = read_numerical_envelope_value(txt, env_re, val_re=env_trigger_re)
        patch['env_1_quantize'], patch['env_2_quantize'] = read_numerical_envelope_value(txt, env_re, val_re=env_quantize_re)
        patch['env_1_curve'], patch['env_2_curve'] = read_numerical_envelope_value(txt, env_re, val_re=env_curve_re)
        patch['env_1_release_on'], patch['env_2_release_on'] = read_numerical_envelope_value(txt, env_re, val_re=env_release_on_re)
        patch['env_1_key_follow'], patch['env_2_key_follow'] = read_numerical_envelope_value(txt, env_re, val_re=env_key_follow_re)

        patches.append(patch)

    # save dataset as dataframe
    df = pd.DataFrame(patches)

    # remove duplicates (only based on preset name)
    df_unique = remove_duplicates(df)

    # get statistics for dataset
    stats = df_unique.describe()

    # seperate numeric and non numeric columns
    non_numeric_cols = ['meta_name', 'meta_location', 'tags_categories', 'tags_features', 'tags_character']
    numeric_cols = [c for c in df_unique.columns if c not in non_numeric_cols]

    # normalize numeric columns
    normalize_columns(df_unique, numeric_cols)

    # save dataframe + stats
    df_unique.to_parquet('data/dataset.parquet', compression='gzip')
    stats.to_csv('data/dataset_stats.csv')