from config import DIVA_PRESET_DIR
from utils import get_all_preset_files, convert_fxp_to_h2p

def main():
    """
    does the following preset conversion operations:
    - .fxp to .h2p
    """

    fxp_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR, file_type='fxp')

    convert_fxp_to_h2p(fxp_files)