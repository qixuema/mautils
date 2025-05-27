# utils/logging_config.py

import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbose output
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
