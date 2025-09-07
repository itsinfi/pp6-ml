import dawdreamer as daw
from config import SAMPLE_RATE, BLOCK_SIZE, NOTE_PITCH, NOTE_VELOCITY, NOTE_DURATION

def init_dawdreamer(sample_rate: int = SAMPLE_RATE):
    # create a render engine
    engine = daw.RenderEngine(sample_rate, BLOCK_SIZE)

    # create a plugin processor for diva
    diva = engine.make_plugin_processor('diva', r'C:\Program Files\Common Files\VST3\Diva(x64).vst3')

    # print infos for diva
    # print('inputs:', diva.get_num_input_channels())
    # print('outputs:', diva.get_num_output_channels())
    # print(diva.get_parameters_description())

    # save parameter description as text file
    # with open('data/parameter_description.txt', mode='w', encoding='utf-8') as f:
    #     f.writelines(str(desc) + '\n' for desc in diva.get_parameters_description())

    # midi configuration
    diva.add_midi_note(note=NOTE_PITCH, velocity=NOTE_VELOCITY, start_time=0, duration=NOTE_DURATION)

    # graph configuration
    engine.load_graph([(diva, [])])

    return engine, diva