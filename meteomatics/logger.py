import logging
import sys


def create_log_handler(
    fmt='%(asctime)s| %(levelname)s |%(message)s',
    stream=sys.stderr,
):
    formatter = logging.Formatter(fmt)
    
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(formatter)
    
    logging.getLogger("meteomatics").addHandler(handler)
