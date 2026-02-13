# Forecasting - MLOps Pipeline

Pipeline MLOps completo para forecasting de ventas con simulación de escenarios.

## Estructura

```
FORECATING/
├── src/
│   ├── config.py              # Configuración centralizada
│   ├── data/                 # ETL: extract, transform, load
│   ├── pipeline/             # Training & Inference pipelines
│   └── utils/                # Logger, validators
├── app/App.py               # Streamlit dashboard
├── main.py                  # CLI de orquestación
├── models/                  # Modelos entrenados
└── data/                    # Datos raw y processed
```

## Uso

```bash
# ETL - Extraer, transformar, cargar datos
python main.py etl

# Entrenar modelo
python main.py train

# Hacer predicciones
python main.py infer

# Ejecutar todo el pipeline
python main.py full

# Ejecutar app Streamlit
streamlit run app/App.py
```

## Métricas del Modelo

| Métrica | Train | Test |
|---------|-------|------|
| RMSE | 0.76 | 1.24 |
| R² | 0.984 | 0.970 |
