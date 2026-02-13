"""Transform module - Feature engineering."""

import pandas as pd
import numpy as np
import holidays
from typing import List, Dict, Optional, Tuple

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def crear_features_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de tiempo: año, mes, día de la semana, etc.

    Args:
        df: DataFrame con columna de fecha

    Returns:
        DataFrame con features de tiempo
    """
    df = df.copy()

    if config.data.fecha_col not in df.columns:
        raise ValueError(f"Columna '{config.data.fecha_col}' no encontrada")

    if not pd.api.types.is_datetime64_any_dtype(df[config.data.fecha_col]):
        df[config.data.fecha_col] = pd.to_datetime(df[config.data.fecha_col])

    df[config.features.calendar_features[0]] = df[config.data.fecha_col].dt.year
    df[config.features.calendar_features[1]] = df[config.data.fecha_col].dt.month
    df[config.features.calendar_features[2]] = df[config.data.fecha_col].dt.weekday
    df[config.features.calendar_features[3]] = df[config.data.fecha_col].dt.day

    df[config.features.calendar_features[4]] = (
        df[config.features.calendar_features[2]].isin([5, 6]).astype(int)
    )

    logger.debug(f"Features de tiempo creadas: {config.features.calendar_features}")

    return df


def crear_features_festivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features para festividades y eventos especiales.

    Args:
        df: DataFrame con columna de fecha

    Returns:
        DataFrame con features de festividades
    """
    df = df.copy()

    cal_es = holidays.country_holidays(config.features.holiday_country)
    df["es_festivo"] = df[config.data.fecha_col].isin(cal_es).astype(int)

    def get_black_friday(year: int) -> pd.Timestamp:
        last_november = pd.Timestamp(
            year=year, month=config.features.black_friday_month, day=30
        )
        offset = (last_november.weekday() - 4) % 7
        return last_november - pd.Timedelta(days=offset)

    years = df[config.features.calendar_features[0]].unique()
    black_fridays = [get_black_friday(y) for y in years]

    df["es_black_friday"] = df[config.data.fecha_col].isin(black_fridays).astype(int)

    logger.debug("Features de festividades creadas: es_festivo, es_black_friday")

    return df


def crear_features_lags(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    producto_col: str = "producto_id",
) -> pd.DataFrame:
    """
    Crea features de lags (ventanas temporales anteriores).

    Args:
        df: DataFrame con datos de ventas
        lags: Lista de períodos de lag (default: config.features.lag_periods)
        producto_col: Columna que identifica el producto

    Returns:
        DataFrame con features de lags
    """
    df = df.copy()

    if lags is None:
        lags = config.features.lag_periods

    df = df.sort_values([producto_col, config.data.fecha_col])

    target = config.data.target_col

    for lag in lags:
        col_name = f"unidades_L{lag}"
        df[col_name] = df.groupby(producto_col)[target].shift(lag)
        df[col_name] = df[col_name].fillna(0)

    for window in config.features.rolling_windows:
        col_name = f"MM{window}"
        df[col_name] = df.groupby(producto_col)[target].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[col_name] = df[col_name].fillna(0)

    logger.debug(
        f"Features de lags creadas: {[f'unidades_L{l}' for l in lags] + [f'MM{w}' for w in config.features.rolling_windows]}"
    )

    return df


def crear_dummies(
    df: pd.DataFrame, columnas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Crea variables dummy (one-hot encoding) para categorías.

    Args:
        df: DataFrame con columnas categóricas
        columnas: Lista de columnas a codificar (default: config.data.categoria_cols)

    Returns:
        DataFrame con dummies
    """
    df = df.copy()

    if columnas is None:
        columnas = config.data.categoria_cols

    cols_existentes = [c for c in columnas if c in df.columns]

    dummies_dict = {}

    for col in cols_existentes:
        prefix_map = {
            "nombre": "nombre_h",
            "categoria": "categoria_h",
            "subcategoria": "subcategoria_h",
        }
        prefix = prefix_map.get(col, col[:3] + "_h")
        dummies = pd.get_dummies(df[col], prefix=prefix)
        dummies_dict[col] = dummies

    df = pd.concat([df] + list(dummies_dict.values()), axis=1)

    total_dummies = sum(len(d) for d in dummies_dict.values())
    logger.debug(
        f"Dummies creadas: {total_dummies} columnas para {len(cols_existentes)} categorías"
    )

    return df


def preparar_datos_entrenamiento(
    df: pd.DataFrame, drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepara los datos para entrenamiento:
    - Convierte tipos de datos
    - Elimina columnas no necesarias
    - Retorna lista de features

    Args:
        df: DataFrame transformado
        drop_cols: Columnas a eliminar (opcional)

    Returns:
        Tuple de (DataFrame preparado, lista de features)
    """
    df = df.copy()

    if drop_cols is None:
        drop_cols = [
            config.data.fecha_col,
            config.data.producto_id_col,
            config.data.categoria_cols[0],
            config.data.categoria_cols[1],
            config.data.categoria_cols[2],
        ]

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    feature_cols = [
        c for c in df.columns if c not in cols_to_drop and c != config.data.target_col
    ]

    df_model = df[feature_cols + [config.data.target_col]].copy()

    for col in df_model.select_dtypes(include=["bool"]).columns:
        df_model[col] = df_model[col].astype(int)

    logger.info(f"Datos preparados para entrenamiento: {len(feature_cols)} features")

    return df_model, feature_cols


def transformar_datos(
    df: pd.DataFrame, incluir_lags: bool = True, incluir_dummies: bool = True
) -> pd.DataFrame:
    """
    Aplica todas las transformaciones a los datos.

    Args:
        df: DataFrame raw
        incluir_lags: Si True, crea features de lags
        incluir_dummies: Si True, crea dummies

    Returns:
        DataFrame transformado
    """
    logger.info("Iniciando transformación de datos...")

    df = crear_features_tiempo(df)
    logger.debug("Features de tiempo creadas")

    df = crear_features_festivos(df)
    logger.debug("Features de festividades creadas")

    if incluir_lags:
        df = crear_features_lags(df)
        logger.debug("Features de lags creadas")

    if incluir_dummies:
        df = crear_dummies(df)
        logger.debug("Dummies creadas")

    logger.info(f"Transformación completada: {df.shape[1]} columnas")

    return df
