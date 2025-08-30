from dotenv import load_dotenv
import os

load_dotenv()

DIVA_PRESET_DIR: str = str(os.getenv('DIVA_PRESET_DIR'))
SAMPLE_RATE: int = int(os.getenv('SAMPLE_RATE'))
BLOCK_SIZE: int = int(os.getenv('BLOCK_SIZE'))
NOTE_PITCH: int = int(os.getenv('NOTE_PITCH'))
NOTE_VELOCITY: int = int(os.getenv('NOTE_VELOCITY'))
NOTE_DURATION: float = float(os.getenv('NOTE_DURATION'))
RENDER_DURATION: float = float(os.getenv('RENDER_DURATION'))
FADE_IN_DURATION: float = float(os.getenv('FADE_IN_DURATION'))
FADE_OUT_DURATION: float = float(os.getenv('FADE_OUT_DURATION'))