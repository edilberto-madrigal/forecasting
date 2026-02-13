"""Logger utility - Logging centralizado."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config import config


def setup_logger(
    name: str = "forecasting", level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configura un logger con formato y handlers.

    Args:
        name: Nombre del logger
        level: Nivel de logging
        log_file: Ruta del archivo de log (opcional)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logs_dir = config.paths.logs
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Obtiene un logger configurado.

    Args:
        name: Nombre del logger

    Returns:
        Logger
    """
    root_logger = logging.getLogger("forecasting")

    if not root_logger.handlers:
        setup_logger()

    return logging.getLogger(name)
