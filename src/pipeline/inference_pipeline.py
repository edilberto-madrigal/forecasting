"""Inference Pipeline - Predicciones en producción."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import joblib

from ..config import config
from ..utils.logger import get_logger
from ..data.transform import transformar_datos

logger = get_logger(__name__)


class InferencePipeline:
    """
    Pipeline de inferencia para predicciones en producción.
    Maneja predicciones recursivas actualizando lags y medias móviles.
    """

    def __init__(self, modelo_path: str = None):
        """
        Inicializa el pipeline de inferencia.

        Args:
            modelo_path: Ruta al modelo guardado (opcional)
        """
        self.modelo_path = modelo_path
        self.pipeline = None
        self.feature_cols = None
        self.lag_cols = []
        self.ma_col = None

        if modelo_path:
            self.cargar_modelo(modelo_path)

    def cargar_modelo(self, filepath: str) -> None:
        """
        Carga el modelo entrenado.

        Args:
            filepath: Ruta al archivo del modelo
        """
        save_dict = joblib.load(filepath)

        self.pipeline = save_dict["pipeline"]
        self.feature_cols = save_dict["feature_cols"]

        self.lag_cols = [c for c in self.feature_cols if "unidades_L" in c]
        self.lag_cols = sorted(self.lag_cols, key=lambda x: int(x.split("_L")[1]))

        self.ma_col = next((c for c in self.feature_cols if c.startswith("MM")), None)

        logger.info(f"Modelo cargado desde: {filepath}")
        logger.info(
            f"Features: {len(self.feature_cols)}, Lags: {self.lag_cols}, MA: {self.ma_col}"
        )

    def _actualizar_lags(
        self,
        fila: pd.Series,
        predicciones_previos: List[float],
        lags_anteriores: List[pd.Series],
    ) -> pd.Series:
        """
        Actualiza los valores de lags con las predicciones previas.

        Args:
            fila: Fila actual del DataFrame
            predicciones_previos: Lista de predicciones anteriores
            lags_anteriores: Valores de lags del día anterior

        Returns:
            Fila con lags actualizados
        """
        fila = fila.copy()

        if self.lag_cols and predicciones_previos:
            primer_lag = self.lag_cols[0]
            fila[primer_lag] = predicciones_previos[-1]

            for i in range(1, len(self.lag_cols)):
                col_actual = self.lag_cols[i]
                col_anterior = self.lag_cols[i - 1]
                if lags_anteriores:
                    fila[col_actual] = lags_anteriores[-1].get(col_anterior, 0)

        if self.ma_col and len(predicciones_previos) >= 7:
            ventana = predicciones_previos[-7:]
            fila[self.ma_col] = np.mean(ventana)

        return fila

    def predecir(
        self, df: pd.DataFrame, recursive: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Realiza predicciones sobre los datos de inferencia.

        Args:
            df: DataFrame con datos de inferencia
            recursive: Si True, actualiza lags recursivamente

        Returns:
            Tuple de (DataFrame con predicciones, métricas)
        """
        if self.pipeline is None:
            raise ValueError("Modelo no cargado. Ejecuta cargar_modelo() primero.")

        df = df.copy()
        df = df.sort_values(config.data.fecha_col).reset_index(drop=True)

        df["unidades_predichas"] = 0.0

        predicciones = []
        lags_anteriores = []

        for idx in range(len(df)):
            fila = df.loc[idx].copy()

            if recursive and idx > 0 and predicciones:
                fila = self._actualizar_lags(fila, predicciones, lags_anteriores)

            X = fila[self.feature_cols].to_frame().T

            y_hat = float(self.pipeline.predict(X)[0])
            y_hat = max(0, y_hat)

            predicciones.append(y_hat)
            lags_anteriores.append(
                fila[self.lag_cols] if self.lag_cols else pd.Series()
            )

            df.loc[idx, "unidades_predichas"] = y_hat

        df["ingresos_proyectados"] = df["precio_venta"] * df["unidades_predichas"]

        kpis = {
            "unidades_totales": float(df["unidades_predichas"].sum()),
            "ingresos_totales": float(df["ingresos_proyectados"].sum()),
            "precio_promedio": float(df["precio_venta"].mean()),
        }

        logger.info(
            f"Predicciones completadas: {kpis['unidades_totales']:.0f} unidades"
        )

        return df, kpis

    def predecir_producto(
        self,
        df_producto: pd.DataFrame,
        descuento_pct: float = 0.0,
        ajuste_competencia_pct: float = 0.0,
        recursive: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Realiza predicciones para un producto específico con ajustes.

        Args:
            df_producto: DataFrame del producto
            descuento_pct: Porcentaje de descuento
            ajuste_competencia_pct: Ajuste de precios de competencia
            recursive: Si True, actualiza lags recursivamente

        Returns:
            Tuple de (DataFrame con predicciones, KPIs)
        """
        df = df_producto.copy()

        factor_descuento = 1.0 - descuento_pct / 100.0
        df["precio_venta"] = df["precio_base"] * factor_descuento

        factor_competencia = 1.0 + ajuste_competencia_pct / 100.0
        for col in ["Amazon", "Decathlon", "Deporvillage"]:
            if col in df.columns:
                df[col] = df[col] * factor_competencia

        if all(c in df.columns for c in ["Amazon", "Decathlon", "Deporvillage"]):
            df["precio_competencia"] = df[["Amazon", "Decathlon", "Deporvillage"]].mean(
                axis=1
            )
        else:
            df["precio_competencia"] = df["precio_venta"]

        df["descuento_porcentaje"] = descuento_pct
        df["ratio_precio"] = df["precio_venta"] / df["precio_competencia"].replace(
            0, np.nan
        )
        df["ratio_precio"] = df["ratio_precio"].fillna(1.0)

        df_resultado, kpis = self.predecir(df, recursive)

        kpis["descuento_promedio"] = descuento_pct

        return df_resultado, kpis

    def predecir_multi_escenario(
        self,
        df_producto: pd.DataFrame,
        descuento_pct: float,
        escenarios_competencia: List[float] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, Dict[str, float]]]:
        """
        Realiza predicciones para múltiples escenarios de competencia.

        Args:
            df_producto: DataFrame del producto
            descuento_pct: Porcentaje de descuento
            escenarios_competencia: Lista de ajustes de competencia

        Returns:
            Diccionario con resultados por escenario
        """
        if escenarios_competencia is None:
            escenarios_competencia = [-5.0, 0.0, 5.0]

        resultados = {}

        for ajuste in escenarios_competencia:
            nombre_escenario = f"Competencia {ajuste:+.0f}%"

            df_esc, kpis = self.predecir_producto(
                df_producto,
                descuento_pct=descuento_pct,
                ajuste_competencia_pct=ajuste,
                recursive=True,
            )

            resultados[nombre_escenario] = (df_esc, kpis)

        return resultados
