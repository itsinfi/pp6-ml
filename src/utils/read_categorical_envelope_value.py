from re import Pattern

def read_categorical_envelope_value(txt: str, env_re: Pattern[str], val_re: Pattern[str], encoder: dict):
    val_1 = encoder.copy()
    val_2 = encoder.copy()

    for env_num, env_block in env_re.findall(txt):
        val_match = val_re.search(env_block)
        
        if val_match:
            try:
                if env_num == '1':
                    val = val_match.group(1)
                    val_1[val] = float(val)
                elif env_num == '2':
                    val = val_match.group(1)
                    val_2[val] = float(val)
            
            except ValueError as err:
                print(err)
                pass

    return val_1, val_2