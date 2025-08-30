import laion_clap as lc
from config import CLAP_CKPT_LOCATION

def init_clap():
    # init the model
    clap = lc.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", tmodel="roberta")

    # load the checkpoint (music-specific prefarably)
    clap.load_ckpt(CLAP_CKPT_LOCATION)

    return clap
