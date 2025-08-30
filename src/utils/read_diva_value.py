from typing import Dict

def read_diva_value(lines: str, index: int, group: str, key: str, param_desc: Dict[int, Dict[str, any]]):
    inside_group = False

    desc = param_desc[index]

    for line in lines:
        line.strip()

        if line.startswith('#cm'):
            inside_group = line.lower().startswith(f'#cm={group.lower()}')
            continue
            
        if inside_group and line.lower().startswith(f'{key.lower()}'):
            val = line.split('=', 1)[1].strip('\n')

            if val.find('.') != -1:
                min = float(desc['min'])
                max = float(desc['max'])
                return (float(val) - min) / (max - min)
            
            if key == 'Module':
                if val.startswith("'Chorus"):
                    val = 0
                elif val.startswith("'Phaser"):
                    val = 1
                elif val.startswith("'Plate"):
                    val = 2
                elif val.startswith("'Delay"):
                    val = 3
                elif val.startswith("'Rotary"):
                    val = 4

            if key in ['Oct', 'Voicing', 'MTunN', 'MTunT', 'TPots', 'TransM', 'Rev', 'SkRev']:
                val = int(val) - 1
            elif key == 'Sync' and group.startswith('LFO'):
                val = int(val) + 3
            elif key == 'Trsp':
                val = int(val) + 24
            elif key == 'PFreq':
                val = int(val) + 1

            num_steps = int(desc['numSteps'])
            return int(val) / (num_steps - 1) 
                
    return None