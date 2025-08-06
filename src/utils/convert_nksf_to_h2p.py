import traceback

def convert_nksf_to_h2p(nksf_files):
    for nksf_file in nksf_files:
        h2p_file = nksf_file[:-4] + 'h2p'

        try:
            with open(nksf_file, mode='rb') as f:
                lines = f.readlines()

            search_for = b'/*@Meta\n'

            lines_to_pop = []

            for i, _ in enumerate(lines):
                if lines[i].endswith(search_for):
                    lines[i] = search_for
                    break
                else:
                    lines_to_pop.append(i)

            for i in sorted(lines_to_pop, reverse=True):
                lines.pop(i)

            decoded_lines = []
            for line in lines:
                try:
                    decoded_lines.append(line.decode('utf-8'))
                except Exception as e:
                    print(f'error when decoding {line}: {e}')
                    traceback.print_exc()
            
            with open(h2p_file, mode='w', encoding='utf-8') as f:
                f.write(''.join(decoded_lines))
            
            print(f'Converted {nksf_file} -> {h2p_file}')

        except Exception as e:
            print(f'error when processing {nksf_file}: {e}')
            traceback.print_exc()