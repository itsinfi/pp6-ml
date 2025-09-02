import logging
import sys

def create_logger(level=logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.flush = sys.stdout.flush

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()
    root.addHandler(handler)

    return root

logger = create_logger()
