import traceback

def convert_fxp_to_h2p(fxp_files):
    for fxp_file in fxp_files:
        h2p_file = fxp_file[:-3] + 'h2p'

        try:
            with open(fxp_file, mode='rb') as f:
                lines = f.readlines()

            search_for = b'/*@Meta\n'

            for i, _ in enumerate(lines):
                if lines[i].endswith(search_for):
                    lines[i] = search_for
                    break
                else:
                    lines.pop(i)

            decoded_lines = []
            for line in lines:
                try:
                    decoded_lines.append(line.decode('utf-8'))
                except Exception as e:
                    print(f'error when decoding {line}: {e}')
                    traceback.print_exc()
            
            with open(h2p_file, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(decoded_lines))
            
            print(f'Converted {fxp_file} -> {h2p_file}')

        except Exception as e:
            print(f'error when processing {fxp_file}: {e}')
            traceback.print_exc()