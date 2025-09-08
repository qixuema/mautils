# utils/logging_config.py

import logging
from loguru import logger
import sys

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,  # Change to DEBUG for more verbose output
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     )


def set_logger_debug(on: bool):
    logger.remove()  # 清空旧的 sink，防重复
    level = "DEBUG" if on else "INFO"
    logger.add(sys.stderr, level=level)

def pdbg(msg: str):
    logger.debug(msg)

def pinfo(msg: str):
    logger.info(msg)

def perr(msg: str):
    logger.error(msg)