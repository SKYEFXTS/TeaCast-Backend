"""
Logging Configuration Module
This module provides centralized logging configuration for the entire TeaCast API.
It sets up consistent logging formats, handlers, and levels across all modules.
"""

import logging
import os

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging with proper formatting and handlers.
    
    Args:
        log_level: The logging level (default: logging.INFO)
        log_file: Path to log file (if None, only console logging is used)
    """
    # Create formatter for detailed logging information
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Configure the root logger - this affects all loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if a log file is specified
    if log_file:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)