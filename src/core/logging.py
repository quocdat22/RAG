"""
Structured logging configuration for RAG system.

This module sets up structured JSON logging with appropriate formatters
and handlers for different environments.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Build formatted message
        log_msg = (
            f"{color}[{record.levelname:8}]{reset} "
            f"{timestamp} "
            f"{record.name:20} "
            f"{record.getMessage()}"
        )

        # Add exception if present
        if record.exc_info:
            log_msg += f"\n{self.formatException(record.exc_info)}"

        return log_msg


def setup_logging(
    log_level: str | None = None,
    log_file: Path | None = None,
    json_format: bool = False,
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_format: Use JSON formatting (useful for production)
    """
    # Get log level from settings if not provided
    level = log_level or settings.log_level
    log_level_int = getattr(logging, level.upper())

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_int)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredConsoleFormatter())

    root_logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(JSONFormatter())  # Always use JSON for files
        root_logger.addHandler(file_handler)

    # Suppress verbose logs from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_execution_time(func):
    """Decorator to log function execution time."""
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"{func.__name__} completed in {execution_time:.2f}s",
                extra={"execution_time": execution_time, "function": func.__name__},
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}",
                extra={
                    "execution_time": execution_time,
                    "function": func.__name__,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    return wrapper


# Initialize logging on module import
setup_logging()


__all__ = [
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "log_execution_time",
    "JSONFormatter",
    "ColoredConsoleFormatter",
]
