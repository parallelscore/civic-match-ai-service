# Import the required libraries
import logging

from colorama import init, Fore, Style

init(autoreset=True)


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = Style.DIM + Fore.WHITE
    green = Fore.GREEN
    yellow = Fore.YELLOW
    red = Fore.RED
    format = '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format,
        logging.INFO: green + format,
        logging.WARNING: yellow + format,
        logging.ERROR: red + format,
        logging.CRITICAL: red + format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style='%')
        return formatter.format(record)


def setup_logger(name):
    # Create a logger
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set the logging level

        # Create a console handler using the custom formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger
