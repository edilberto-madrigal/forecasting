"""Validators utility - Validaciones de datos."""

import pandas as pd
from typing import List, Dict, Any, Optional

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def validar_dataframe(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    check_nulls: bool = True,
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Valida un DataFrame.

    Args:
        df: DataFrame a validar
        required_cols: Lista de columnas requeridas
        check_nulls: Verificar valores nulos
        check_duplicates: Verificar duplicados

    Returns:
        Diccionario con resultados de validación
    """
    results = {"valido": True, "errores": [], "warnings": [], "info": {}}

    results["info"]["filas"] = len(df)
    results["info"]["columnas"] = len(df.columns)

    if required_cols:
        cols_faltantes = [c for c in required_cols if c not in df.columns]
        if cols_faltantes:
            results["valido"] = False
            results["errores"].append(f"Columnas faltantes: {cols_faltantes}")

    if check_nulls:
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            null_counts = df[null_cols].isnull().sum().to_dict()
            results["warnings"].append(f"Valores nulos en: {null_counts}")

    if check_duplicates:
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            results["warnings"].append(f"Filas duplicadas: {n_duplicates}")

    if results["errores"]:
        logger.error(f"Validación fallida: {results['errores']}")
    elif results["warnings"]:
        logger.warning(f"Validación con advertencias: {results['warnings']}")
    else:
        logger.info("DataFrame válido")

    return results


def validar_schema(df: pd.DataFrame, schema: Dict[str, str]) -> Dict[str, Any]:
    """
    Valida el schema (tipos de columnas) de un DataFrame.

    Args:
        df: DataFrame a validar
        schema: Diccionario con columna: tipo esperado

    Returns:
        Diccionario con resultados de validación
    """
    results = {"valido": True, "errores": [], "info": {}}

    type_mapping = {
        "datetime": "datetime64",
        "int": "int64",
        "float": "float64",
        "object": "object",
        "bool": "bool",
    }

    for col, expected_type in schema.items():
        if col not in df.columns:
            results["valido"] = False
            results["errores"].append(f"Columna '{col}' no encontrada")
            continue

        actual_type = str(df[col].dtype)
        expected = type_mapping.get(expected_type, expected_type)

        if expected_type == "int" and "int" in actual_type:
            continue
        if expected_type == "float" and "float" in actual_type:
            continue
        if actual_type != expected:
            results["valido"] = False
            results["errores"].append(
                f"Columna '{col}': tipo '{actual_type}', esperado '{expected}'"
            )

    if results["errores"]:
        logger.error(f"Schema validation failed: {results['errores']}")
    else:
        logger.info("Schema válido")

    return results


def validar_fechas(
    df: pd.DataFrame,
    fecha_col: str = "fecha",
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Valida el rango de fechas en un DataFrame.

    Args:
        df: DataFrame con columna de fecha
        fecha_col: Nombre de la columna de fecha
        min_date: Fecha mínima esperada
        max_date: Fecha máxima esperada

    Returns:
        Diccionario con resultados de validación
    """
    results = {"valido": True, "errores": [], "warnings": [], "info": {}}

    if fecha_col not in df.columns:
        results["valido"] = False
        results["errores"].append(f"Columna '{fecha_col}' no encontrada")
        return results

    if not pd.api.types.is_datetime64_any_dtype(df[fecha_col]):
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")

    min_fecha = df[fecha_col].min()
    max_fecha = df[fecha_col].max()

    results["info"]["min_fecha"] = str(min_fecha)
    results["info"]["max_fecha"] = str(max_fecha)

    if min_date and min_fecha < min_date:
        results["valido"] = False
        results["errores"].append(f"Fecha mínima {min_fecha} < {min_date}")

    if max_date and max_fecha > max_date:
        results["valido"] = False
        results["errores"].append(f"Fecha máxima {max_fecha} > {max_date}")

    return results


def validar_modelo_entrenado(pipeline) -> bool:
    """
    Valida que un modelo esté correctamente entrenado.

    Args:
        pipeline: Pipeline de sklearn

    Returns:
        True si es válido
    """
    if pipeline is None:
        logger.error("Pipeline es None")
        return False

    if not hasattr(pipeline, "predict"):
        logger.error("Pipeline no tiene método predict")
        return False

    logger.info("Pipeline validado correctamente")
    return True
