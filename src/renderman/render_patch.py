import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import librenderman as rm

# TODO: move to config
midi_note_pitch = 22050 # in hz
velocity = 127 # between 0 - 127
note_length_seconds = 2
render_length_seconds = 4 

def render_patch(re: rm.RenderEngine):

    # render patch
    re.render_patch(midi_note_pitch, velocity, note_length_seconds, render_length_seconds)

    # read rendered audio
    af = re.get_audio_frames()

    # write to wav file
    # re.write_to_wav('e.wav')

    # TODO: add fadout

    return af