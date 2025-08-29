import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import librenderman as rm

# TODO: move to config
# config
sample_rate = 44100 # in hz
buffer_size = 512 # in number of samples
fft_size = 512 # in number of samples

def create_render_engine():
    # create a render engine
    re = rm.RenderEngine(sample_rate, buffer_size, fft_size)

    # try to load diva
    if not re.load_plugin(r'C:\Program Files\Common Files\VST3\Diva(x64).vst3', 0):
        print('Error: Diva could not be loaded')
        return
    
    # get the current patch
    # patch = re.get_patch()
    # print('patch:', patch)
    
    # get the plugin param description
    # print('description:', re.get_plugin_parameters_description())

    # override plugin parameter
    # re.override_plugin_parameter(2180, 0.27)

    # patch = re.get_patch()
    # print('patch:', patch)

    return re