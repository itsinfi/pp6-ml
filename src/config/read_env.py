from dotenv import load_dotenv
import os

load_dotenv()

DIVA_PRESET_DIR: str = str(os.getenv('DIVA_PRESET_DIR'))
SAMPLE_RATE: int = int(os.getenv('SAMPLE_RATE'))
BUFFER_SIZE: int = int(os.getenv('BUFFER_SIZE'))
FFT_SIZE: int = int(os.getenv('FFT_SIZE'))
MIDI_NOTE_PITCH: int = int(os.getenv('MIDI_NOTE_PITCH'))
VELOCITY: int = int(os.getenv('VELOCITY'))
NOTE_LENGTH_SECONDS: float = float(os.getenv('NOTE_LENGTH_SECONDS'))
RENDER_LENGTH_SECONDS: float = float(os.getenv('RENDER_LENGTH_SECONDS'))
FADE_IN_SECONDS: float = float(os.getenv('FADE_IN_SECONDS'))
FADE_OUT_SECONDS: float = float(os.getenv('FADE_OUT_SECONDS'))