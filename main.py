"""Main CLI - Orquestación del MLOps Pipeline."""

import argparse
import sys
from pathlib import Path

from src.config import config
from src.utils.logger import setup_logger, get_logger
from src.utils.validators import validar_dataframe
from src.data.extract import cargar_y_merge, cargar_datos_inferencia
from src.data.transform import transformar_datos, preparar_datos_entrenamiento
from src.data.load import guardar_datos, guardar_modelo
from src.pipeline.evaluation import guardar_metricas
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.inference_pipeline import InferencePipeline
from src.pipeline.evaluation import evaluar_modelo

logger = get_logger(__name__)


def step_etl(args):
    """Ejecuta el paso ETL: Extract, Transform, Load."""
    logger.info("=" * 50)
    logger.info("EJECUTANDO PIPELINE ETL")
    logger.info("=" * 50)

    folder = args.folder or "entrenamiento"
    logger.info(f"Cargando datos de: {folder}")

    df = cargar_y_merge(folder=folder)

    validation = validar_dataframe(df)
    if not validation["valido"]:
        logger.error(f"Validación fallida: {validation['errores']}")
        return 1

    logger.info("Transformando datos...")
    df_transformed = transformar_datos(df)

    output_file = f"df_{folder}_transformado.csv"
    guardar_datos(df_transformed, output_file)

    logger.info(f"Pipeline ETL completado: {output_file}")
    return 0


def step_train(args):
    """Ejecuta el paso de entrenamiento."""
    logger.info("=" * 50)
    logger.info("EJECUTANDO PIPELINE DE ENTRENAMIENTO")
    logger.info("=" * 50)

    data_file = args.data or "df_entrenamiento_transformado.csv"
    logger.info(f"Cargando datos desde: {data_file}")

    from src.data.load import cargar_datos

    df = cargar_datos(data_file)

    df_model, feature_cols = preparar_datos_entrenamiento(df)

    logger.info(f"Entrenando modelo con {len(feature_cols)} features...")

    pipeline = TrainingPipeline(
        model_params=config.model.params,
        test_size=config.model.test_size,
        random_state=config.random_state,
    )

    pipeline.train(df_model, feature_cols=feature_cols)

    metrics = pipeline.get_metrics()
    logger.info(f"Métricas - Train RMSE: {metrics['train']['rmse']:.4f}")
    logger.info(f"Métricas - Test RMSE: {metrics['test']['rmse']:.4f}")

    model_file = args.output or "modelo_pipeline.joblib"
    pipeline.save(model_file)

    guardar_metricas(metrics)

    logger.info(f"Modelo entrenado y guardado: {model_file}")
    return 0


def step_infer(args):
    """Ejecuta el paso de inferencia."""
    logger.info("=" * 50)
    logger.info("EJECUTANDO PIPELINE DE INFERENCIA")
    logger.info("=" * 50)

    model_file = args.model or "modelo_pipeline.joblib"
    logger.info(f"Cargando modelo desde: {model_file}")

    inference = InferencePipeline(model_file)

    from src.data.load import cargar_datos

    df_inferencia = cargar_datos("df_inferencia_transformado.csv")

    df_resultado, kpis = inference.predecir(df_inferencia)

    logger.info(f"Unidades predichas: {kpis['unidades_totales']:.0f}")
    logger.info(f"Ingresos proyectados: €{kpis['ingresos_totales']:.2f}")

    if args.output:
        guardar_datos(df_resultado, args.output)
        logger.info(f"Predicciones guardadas en: {args.output}")

    return 0


def step_full(args):
    """Ejecuta el pipeline completo: ETL + Train + Infer."""
    logger.info("=" * 50)
    logger.info("EJECUTANDO PIPELINE COMPLETO")
    logger.info("=" * 50)

    logger.info("\n--- Paso ETL ---")
    step_etl(argparse.Namespace(folder="entrenamiento"))

    logger.info("\n--- Paso Train ---")
    step_train(
        argparse.Namespace(
            data="df_entrenamiento_transformado.csv", output="modelo_pipeline.joblib"
        )
    )

    logger.info("\n--- Infer ---")
    step_infer(
        argparse.Namespace(model="modelo_pipeline.joblib", output="predicciones.csv")
    )

    logger.info("Pipeline completo ejecutado exitosamente!")
    return 0


def main():
    """Punto de entrada del CLI."""
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline para Forecasting de Ventas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py etl                              # Ejecutar solo ETL
  python main.py train                            # Entrenar modelo
  python main.py infer                            # Hacer predicciones
  python main.py full                            # Ejecutar todo el pipeline
  python main.py train --output mi_modelo.joblib  # Guardar con nombre personalizado
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    parser_etl = subparsers.add_parser(
        "etl", help="Ejecutar ETL (Extract, Transform, Load)"
    )
    parser_etl.add_argument("--folder", type=str, help="Carpeta de datos raw")

    parser_train = subparsers.add_parser("train", help="Entrenar modelo")
    parser_train.add_argument(
        "--data", type=str, help="Archivo de datos de entrenamiento"
    )
    parser_train.add_argument("--output", type=str, help="Archivo de salida del modelo")

    parser_infer = subparsers.add_parser("infer", help="Hacer predicciones")
    parser_infer.add_argument("--model", type=str, help="Archivo del modelo")
    parser_infer.add_argument(
        "--output", type=str, help="Archivo de salida de predicciones"
    )

    parser_full = subparsers.add_parser("full", help="Ejecutar pipeline completo")

    args = parser.parse_args()

    setup_logger()

    if args.command == "etl":
        return step_etl(args)
    elif args.command == "train":
        return step_train(args)
    elif args.command == "infer":
        return step_infer(args)
    elif args.command == "full":
        return step_full(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
