# -*- coding: utf-8 -*-

import logging
import sys

from ._constants_ import LOGGERNAME

def create_log_handler(fmt='%(asctime)s| %(levelname)s |%(message)s', stream=sys.stderr):
    formatter = logging.Formatter(fmt)

    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(formatter)

    logging.getLogger(LOGGERNAME).addHandler(handler)


def set_log_level(level):
    """Use this function change the internal log level of the Meteomatics API Python connector."""
    if logging.getLogger(LOGGERNAME) is None:
        create_log_handler()

    logging.getLogger(LOGGERNAME).setLevel(level)
