from config import DIVA_PRESET_DIR
from utils import get_all_preset_files
from renderman import create_render_engine, render_patch

def main():
    preset_files = get_all_preset_files(preset_dir=DIVA_PRESET_DIR)

    re = create_render_engine()

    print(render_patch(re))

