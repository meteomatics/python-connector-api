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
    """Use this function to change the internal log level of the Meteomatics API Python connector.

    Note
    ----
    The log level is propagated to the Python logging library and so it's just an integer. Look at the aliases defined
    in the library (https://docs.python.org/3/library/logging.html#logging-levels) for convenience.
    """
    logger = logging.getLogger(LOGGERNAME)

    if not logger.handlers:
        create_log_handler()

    logger.setLevel(level)
