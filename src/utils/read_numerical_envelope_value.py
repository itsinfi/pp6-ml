from typing import Pattern

def read_numerical_envelope_value(txt: str, env_re: Pattern[str], val_re: Pattern[str]):
    val_1 = 0
    val_2 = 0

    for env_num, env_block in env_re.findall(txt):
        val_match = val_re.search(env_block)
        
        if val_match:
            try:
                if env_num == '1':
                    val_1 = float(val_match.group(1))
                elif env_num == '2':
                    val_2 = float(val_match.group(1))
            
            except ValueError as err:
                print(err)
                pass

    return val_1, val_2