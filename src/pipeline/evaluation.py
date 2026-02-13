"""Evaluation module - Métricas y evaluación de modelos."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def evaluar_modelo(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calcula métricas de evaluación del modelo.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Diccionario con métricas
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }

    logger.info(
        f"Métricas - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%"
    )

    return metrics


def guardar_metricas(
    metrics: Dict[str, Any],
    filename: str = "metricas.json",
    base_path: Optional[Path] = None,
) -> Path:
    """
    Guarda las métricas en un archivo JSON.

    Args:
        metrics: Diccionario de métricas
        filename: Nombre del archivo
        base_path: Ruta base del proyecto

    Returns:
        Path del archivo guardado
    """
    if base_path is None:
        base_path = config.paths.root

    metrics_path = base_path / "models"
    metrics_path.mkdir(parents=True, exist_ok=True)

    file_path = metrics_path / filename

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Métricas guardadas en: {file_path}")

    return file_path


def cargar_metricas(
    filename: str = "metricas.json", base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Carga las métricas desde un archivo JSON.

    Args:
        filename: Nombre del archivo
        base_path: Ruta base del proyecto

    Returns:
        Diccionario de métricas
    """
    if base_path is None:
        base_path = config.paths.root

    file_path = base_path / "models" / filename

    if not file_path.exists():
        logger.warning(f"Métricas no encontradas: {file_path}")
        return {}

    with open(file_path, "r") as f:
        metrics = json.load(f)

    logger.info(f"Métricas cargadas desde: {file_path}")

    return metrics


def comparar_modelos(resultados: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compara métricas de múltiples modelos.

    Args:
        resultados: Diccionario con métricas por modelo

    Returns:
        DataFrame comparativo
    """
    df = pd.DataFrame(resultados).T

    df = df.sort_values("rmse")

    logger.info(f"Comparación de {len(resultados)} modelos:")
    logger.info(f"\n{df}")

    return df


def feature_importance_df(importances: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Crea un DataFrame de importancia de features.

    Args:
        importances: Array de importancias
        feature_names: Nombres de features

    Returns:
        DataFrame ordenado por importancia
    """
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

    return df
