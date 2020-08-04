import os
import sys
import logging

def setup_logger(folder=None, name='hiertype', level=logging.INFO):
    """Write logs to the filepath at folder/name.log and also to stdout."""
    if folder is None:
        folder = os.getcwd()
    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, name + '.log')

    #delay is a half-fix for mem leak: https://bugs.python.org/issue23010
    file_handler = logging.FileHandler(fp, delay=True) 
    file_formatter = logging.Formatter('[%(asctime)s:%(levelname)s:%(name)s:%(funcName)s] %(message)s')
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('[%(asctime)s:%(levelname)s:%(name)s] %(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.propagate = False #don't propagate handlers to inheritors 
    return logger