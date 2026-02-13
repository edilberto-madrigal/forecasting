"""Configuración centralizada del proyecto de Forecasting."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathsConfig:
    """Configuración de rutas del proyecto."""

    root: Path = Path(__file__).parent.parent
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    models: Path = field(init=False)
    logs: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self):
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.models = self.root / "models"
        self.logs = self.root / "logs"
        self.reports = self.root / "notebooks" / "reports"

        for path in [self.data_processed, self.models, self.logs, self.reports]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuración de datos."""

    ventas_file: str = "ventas.csv"
    competencia_file: str = "competencia.csv"
    inferencia_file: str = "ventas_2025_inferencia.csv"

    fecha_col: str = "fecha"
    producto_id_col: str = "producto_id"
    target_col: str = "unidades_vendidas"

    categoria_cols: List[str] = field(
        default_factory=lambda: ["nombre", "categoria", "subcategoria"]
    )

    numeric_cols: List[str] = field(
        default_factory=lambda: [
            "precio_base",
            "precio_venta",
            "ingresos",
            "Amazon",
            "Decathlon",
            "Deporvillage",
        ]
    )


@dataclass
class FeatureConfig:
    """Configuración de features."""

    lag_periods: List[int] = field(default_factory=lambda: [1, 2])
    rolling_windows: List[int] = field(default_factory=lambda: [7])

    calendar_features: List[str] = field(
        default_factory=lambda: ["año", "mes", "dia_semana", "dia_mes", "es_fin_semana"]
    )

    holiday_country: str = "ES"
    black_friday_month: int = 11
    black_friday_day: int = 28


@dataclass
class ModelConfig:
    """Configuración del modelo."""

    model_type: str = "HistGradientBoostingRegressor"

    params: dict = field(
        default_factory=lambda: {
            "learning_rate": 0.1,
            "max_iter": 200,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "random_state": 42,
        }
    )

    cv_folds: int = 5
    test_size: float = 0.2


@dataclass
class AppConfig:
    """Configuración de la aplicación Streamlit."""

    page_title: str = "Simulador de Ventas - Noviembre 2025"
    page_icon: str = "📈"
    layout: str = "wide"

    productos_default: Optional[List[str]] = None

    descuento_min: int = -50
    descuento_max: int = 50
    descuento_step: int = 5
    descuento_default: int = 0


@dataclass
class Config:
    """Configuración global del proyecto."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    app: AppConfig = field(default_factory=AppConfig)

    verbose: bool = True
    random_state: int = 42


config = Config()
