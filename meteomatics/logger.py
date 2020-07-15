import logging
import sys


def create_log_handler(fmt='%(asctime)s| %(levelname)s |%(message)s'):
    formatter = logging.Formatter(fmt)
    
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)
    
    logging.getLogger("meteomatics").addHandler(handler)
