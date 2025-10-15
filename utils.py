import logging
import os


def setup_logger(name, log_file, level=logging.INFO):

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory: {e}")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger