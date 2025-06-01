import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

LOG_FILE_PATH = os.path.join(LOGS_DIR, "trading_system.log")

def setup_logger(name="trading_system", log_level=logging.INFO):
    """Sets up a logger that logs to both console and a file."""
    logger = logging.getLogger(name)
    if logger.hasHandlers(): # Avoid adding multiple handlers if already configured
        return logger

    logger.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Rotating)
    fh = RotatingFileHandler(LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5) # 10MB per file, 5 backups
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Example usage (optional, can be removed or commented out)
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger setup complete. This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
