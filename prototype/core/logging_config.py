import logging
import sys

def configure_logging(level=logging.INFO):
    """
    Configure structured logging for Spectra.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger("spectra")
    root_logger.setLevel(level)
    
    # Avoid duplicate handlers if called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)
        
    return root_logger

# Initialize default logger
logger = configure_logging()
