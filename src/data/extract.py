"""Extract module - Cargar datos raw."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def cargar_datos_raw(
    folder: str = "entrenamiento", base_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos crudos de ventas y competencia.

    Args:
        folder: Subcarpeta dentro de data/raw (default: 'entrenamiento')
        base_path: Ruta base del proyecto (opcional)

    Returns:
        Tuple de (df_ventas, df_competencia)
    """
    if base_path is None:
        base_path = config.paths.root

    data_path = base_path / "data" / "raw" / folder

    ventas_path = data_path / config.data.ventas_file
    competencia_path = data_path / config.data.competencia_file

    logger.info(f"Cargando datos de: {data_path}")

    df_ventas = pd.read_csv(ventas_path)
    logger.info(
        f"Ventas cargadas: {df_ventas.shape[0]} filas, {df_ventas.shape[1]} columnas"
    )

    df_competencia = pd.read_csv(competencia_path)
    logger.info(
        f"Competencia cargada: {df_competencia.shape[0]} filas, {df_competencia.shape[1]} columnas"
    )

    return df_ventas, df_competencia


def cargar_datos_inferencia(
    file: Optional[str] = None, base_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Carga los datos de inferencia.

    Args:
        file: Nombre del archivo de inferencia (default: config.data.inferencia_file)
        base_path: Ruta base del proyecto (opcional)

    Returns:
        DataFrame de inferencia
    """
    if base_path is None:
        base_path = config.paths.root

    if file is None:
        file = config.data.inferencia_file

    data_path = base_path / "data" / "raw" / "inferencia" / file

    logger.info(f"Cargando datos de inferencia: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Inferencia cargada: {df.shape[0]} filas, {df.shape[1]} columnas")

    return df


def merge_datos(df_ventas: pd.DataFrame, df_competencia: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona los DataFrames de ventas y competencia.

    Args:
        df_ventas: DataFrame de ventas
        df_competencia: DataFrame de competencia

    Returns:
        DataFrame fusionado
    """
    df_ventas[config.data.fecha_col] = pd.to_datetime(df_ventas[config.data.fecha_col])
    df_competencia[config.data.fecha_col] = pd.to_datetime(
        df_competencia[config.data.fecha_col]
    )

    df = pd.merge(
        df_ventas,
        df_competencia,
        on=[config.data.fecha_col, config.data.producto_id_col],
        how="left",
    )

    logger.info(f"Datos fusionados: {df.shape[0]} filas, {df.shape[1]} columnas")

    return df


def cargar_y_merge(
    folder: str = "entrenamiento", base_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Carga y fusiona los datos crudos en un solo DataFrame.

    Args:
        folder: Subcarpeta dentro de data/raw
        base_path: Ruta base del proyecto (opcional)

    Returns:
        DataFrame combinado
    """
    df_ventas, df_competencia = cargar_datos_raw(folder, base_path)
    df = merge_datos(df_ventas, df_competencia)

    return df
