import shutil
import os
from itertools import cycle
import torch
import logging.config
from datetime import datetime
import json

def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
