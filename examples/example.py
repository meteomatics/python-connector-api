#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:=utf-8

import argparse
import logging
import sys
from credentials import username as username_default, password as password_default

from meteomatics.logger import create_log_handler
from meteomatics._constants_ import LOGGERNAME


def run_example(example_lambda):
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default=username_default)
    parser.add_argument('--password', default=password_default)
    arguments = parser.parse_args()

    username = arguments.username
    password = arguments.password

    create_log_handler()
    logging.getLogger(LOGGERNAME).setLevel(logging.INFO)
    _logger = logging.getLogger(LOGGERNAME)

    if username is None or password is None:
        _logger.info(
        "You need to provide a username and a password, either on the command line or by inserting them in the script")
        sys.exit()

    example_lambda(username, password, _logger)
