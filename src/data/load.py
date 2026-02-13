"""Load module - Guardar y cargar datos procesados."""

import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def guardar_datos(
    df: pd.DataFrame,
    filename: str,
    folder: str = "processed",
    base_path: Optional[Path] = None,
) -> Path:
    """
    Guarda un DataFrame en formato CSV.

    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo
        folder: Subcarpeta dentro de data (default: 'processed')
        base_path: Ruta base del proyecto (opcional)

    Returns:
        Path del archivo guardado
    """
    if base_path is None:
        base_path = config.paths.root

    output_path = base_path / "data" / folder
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename

    df.to_csv(file_path, index=False)
    logger.info(f"Datos guardados en: {file_path}")

    return file_path


def cargar_datos(
    filename: str, folder: str = "processed", base_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Carga un DataFrame desde CSV.

    Args:
        filename: Nombre del archivo
        folder: Subcarpeta dentro de data (default: 'processed')
        base_path: Ruta base del proyecto (opcional)

    Returns:
        DataFrame cargado
    """
    if base_path is None:
        base_path = config.paths.root

    file_path = base_path / "data" / folder / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Datos cargados desde: {file_path}")

    return df


def guardar_modelo(
    modelo: Any, filename: str = "modelo_final.joblib", base_path: Optional[Path] = None
) -> Path:
    """
    Guarda un modelo usando joblib.

    Args:
        modelo: Modelo a guardar
        filename: Nombre del archivo
        base_path: Ruta base del proyecto (opcional)

    Returns:
        Path del archivo guardado
    """
    if base_path is None:
        base_path = config.paths.root

    models_path = base_path / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    file_path = models_path / filename

    joblib.dump(modelo, file_path)
    logger.info(f"Modelo guardado en: {file_path}")

    return file_path


def cargar_modelo(
    filename: str = "modelo_final.joblib", base_path: Optional[Path] = None
) -> Any:
    """
    Carga un modelo desde joblib.

    Args:
        filename: Nombre del archivo
        base_path: Ruta base del proyecto (opcional)

    Returns:
        Modelo cargado
    """
    if base_path is None:
        base_path = config.paths.root

    file_path = base_path / "models" / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {file_path}")

    modelo = joblib.load(file_path)
    logger.info(f"Modelo cargado desde: {file_path}")

    return modelo


def guardar_config(
    config_dict: Dict[str, Any],
    filename: str = "config.joblib",
    base_path: Optional[Path] = None,
) -> Path:
    """
    Guarda la configuración del pipeline.

    Args:
        config_dict: Diccionario de configuración
        filename: Nombre del archivo
        base_path: Ruta base del proyecto (opcional)

    Returns:
        Path del archivo guardado
    """
    if base_path is None:
        base_path = config.paths.root

    config_path = base_path / "models"
    config_path.mkdir(parents=True, exist_ok=True)

    file_path = config_path / filename

    joblib.dump(config_dict, file_path)
    logger.info(f"Configuración guardada en: {file_path}")

    return file_path


def cargar_config(
    filename: str = "config.joblib", base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Carga la configuración del pipeline.

    Args:
        filename: Nombre del archivo
        base_path: Ruta base del proyecto (opcional)

    Returns:
        Diccionario de configuración
    """
    if base_path is None:
        base_path = config.paths.root

    file_path = base_path / "models" / filename

    if not file_path.exists():
        logger.warning(f"Configuración no encontrada: {file_path}")
        return {}

    config_dict = joblib.load(file_path)
    logger.info(f"Configuración cargada desde: {file_path}")

    return config_dict
