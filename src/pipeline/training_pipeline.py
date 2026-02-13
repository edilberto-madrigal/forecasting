"""Training Pipeline - sklearn Pipeline para entrenamiento."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict, Any, Optional, List
import joblib

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Pipeline de entrenamiento completo para forecasting de ventas.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Inicializa el pipeline de entrenamiento.

        Args:
            model_params: Parámetros del modelo (default: config.model.params)
            test_size: Proporción para test
            random_state: Semilla aleatoria
        """
        self.model_params = model_params or config.model.params
        self.test_size = test_size
        self.random_state = random_state

        self.pipeline = None
        self.feature_cols = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.metrics = {}

    def _get_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identifica columnas numéricas y categóricas.

        Args:
            df: DataFrame de entrenamiento

        Returns:
            Tuple de (columnas_numéricas, columnas_categóricas)
        """
        exclude_cols = [
            config.data.target_col,
            config.data.fecha_col,
            config.data.producto_id_col,
        ]

        numeric = []
        categorical = []

        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                if df[col].nunique() > 10 or col.startswith(
                    ("precio", "Amazon", "Decathlon", "Deporvillage", "unidades_", "MM")
                ):
                    numeric.append(col)
                else:
                    categorical.append(col)
            elif df[col].dtype == "object" or df[col].dtype.name == "category":
                categorical.append(col)

        logger.debug(
            f"Columnas numéricas: {len(numeric)}, categóricas: {len(categorical)}"
        )

        return numeric, categorical

    def _build_pipeline(
        self, numeric_cols: List[str], categorical_cols: List[str]
    ) -> Pipeline:
        """
        Construye el sklearn Pipeline.

        Args:
            numeric_cols: Lista de columnas numéricas
            categorical_cols: Lista de columnas categóricas

        Returns:
            Pipeline de sklearn
        """
        transformers = []

        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))

        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", HistGradientBoostingRegressor(**self.model_params)),
            ]
        )

        logger.info("Pipeline sklearn construido")

        return pipeline

    def train(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "TrainingPipeline":
        """
        Entrena el modelo.

        Args:
            df: DataFrame con features y target
            target_col: Nombre de la columna objetivo
            feature_cols: Lista de columnas a usar como features

        Returns:
            self
        """
        target_col = target_col or config.data.target_col

        if feature_cols is None:
            self.numeric_cols, self.categorical_cols = self._get_column_types(df)
            feature_cols = self.numeric_cols + self.categorical_cols
        else:
            self.feature_cols = feature_cols
            df_subset = df[feature_cols]
            self.numeric_cols, self.categorical_cols = self._get_column_types(df_subset)

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(
            f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras"
        )

        self.pipeline = self._build_pipeline(self.numeric_cols, self.categorical_cols)

        logger.info("Entrenando modelo...")
        self.pipeline.fit(X_train, y_train)

        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)

        self.metrics = {
            "train": {
                "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "mae": mean_absolute_error(y_train, y_pred_train),
                "r2": r2_score(y_train, y_pred_train),
            },
            "test": {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "mae": mean_absolute_error(y_test, y_pred_test),
                "r2": r2_score(y_test, y_pred_test),
            },
        }

        logger.info(
            f"Métricas - Train RMSE: {self.metrics['train']['rmse']:.4f}, "
            f"Test RMSE: {self.metrics['test']['rmse']:.4f}"
        )

        self.feature_cols = feature_cols

        return self

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> Dict[str, float]:
        """
        Realiza cross-validation.

        Args:
            df: DataFrame de datos
            cv: Número de folds

        Returns:
            Diccionario con métricas de CV
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Ejecuta train() primero.")

        X = df[self.feature_cols]
        y = df[config.data.target_col]

        scores = cross_val_score(
            self.pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error"
        )

        cv_metrics = {"cv_rmse_mean": -scores.mean(), "cv_rmse_std": scores.std()}

        logger.info(
            f"Cross-validation RMSE: {-scores.mean():.4f} (+/- {scores.std():.4f})"
        )

        return cv_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones.

        Args:
            X: DataFrame con features

        Returns:
            Array de predicciones
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Ejecuta train() primero.")

        return self.pipeline.predict(X)

    def save(self, filepath: str) -> None:
        """
        Guarda el pipeline entrenado.

        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Ejecuta train() primero.")

        save_dict = {
            "pipeline": self.pipeline,
            "feature_cols": self.feature_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "metrics": self.metrics,
            "model_params": self.model_params,
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"Modelo guardado en: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TrainingPipeline":
        """
        Carga un pipeline entrenado.

        Args:
            filepath: Ruta del modelo guardado

        Returns:
            Instancia de TrainingPipeline con el modelo cargado
        """
        save_dict = joblib.load(filepath)

        instance = cls(model_params=save_dict.get("model_params"))
        instance.pipeline = save_dict["pipeline"]
        instance.feature_cols = save_dict["feature_cols"]
        instance.numeric_cols = save_dict["numeric_cols"]
        instance.categorical_cols = save_dict["categorical_cols"]
        instance.metrics = save_dict.get("metrics", {})

        logger.info(f"Modelo cargado desde: {filepath}")

        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene importancia de features.

        Returns:
            DataFrame con importancia de features
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no entrenado. Ejecuta train() primero.")

        regressor = self.pipeline.named_steps["regressor"]

        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_

            feature_names = self.feature_cols

            df_importance = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            return df_importance

        return pd.DataFrame()

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna las métricas del entrenamiento.

        Returns:
            Diccionario de métricas
        """
        return self.metrics
