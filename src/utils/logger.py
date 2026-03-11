"""Centralised logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger as _logger

_CONFIGURED = False
_LOG_DIR = Path("outputs/logs")


def get_logger(name: str = "robot_arm"):
    """Return a configured loguru logger bound with the given name.

    On the first call the logger is configured with:
    - A coloured stderr sink at DEBUG level.
    - A rotating file sink inside ``outputs/logs/``.

    Args:
        name: Module or component name to bind to the logger.

    Returns:
        A loguru ``BoundLogger`` instance.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        _logger.remove()
        _logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[name]}</cyan> - <level>{message}</level>"
            ),
            colorize=True,
        )
        _logger.add(
            _LOG_DIR / "robot_arm_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} - {message}",
        )
        _CONFIGURED = True

    return _logger.bind(name=name)
