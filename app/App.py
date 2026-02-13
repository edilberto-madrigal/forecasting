"""Streamlit App - Simulador de Ventas con MLOps Pipeline."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.inference_pipeline import InferencePipeline
from src.data.load import cargar_datos


@st.cache_data
def cargar_datos_inferencia(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {path_csv}")
    df = pd.read_csv(path_csv)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def formatear_euros(valor: float) -> str:
    return f"€{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def formatear_unidades(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def crear_grafico_predicciones(
    df_resultado: pd.DataFrame, nombre_producto: str
) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=df_resultado,
        x="fecha",
        y="unidades_predichas",
        marker="o",
        color="#667eea",
        ax=ax,
    )

    bf_mask = (df_resultado["mes"] == 11) & (df_resultado["dia_mes"] == 28)
    if bf_mask.any():
        fila_bf = df_resultado.loc[bf_mask].iloc[0]
        ax.axvline(fila_bf["fecha"], color="#ff6b6b", linestyle="--", linewidth=1.5)
        ax.scatter(
            [fila_bf["fecha"]],
            [fila_bf["unidades_predichas"]],
            color="red",
            s=80,
            zorder=5,
        )
        ax.annotate(
            "Black Friday",
            xy=(fila_bf["fecha"], fila_bf["unidades_predichas"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="#ff6b6b",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ff6b6b"),
        )

    ax.set_title(
        f"Predicción diaria de unidades vendidas - {nombre_producto} (Noviembre 2025)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades predichas")

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def preparar_tabla_detalle(df_resultado: pd.DataFrame) -> pd.DataFrame:
    df_tabla = df_resultado.copy()
    df_tabla["fecha"] = df_tabla["fecha"].dt.date

    mapa_dia_semana = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }
    df_tabla["dia_semana_nombre"] = df_tabla["dia_semana"].map(mapa_dia_semana)

    df_tabla["precio_venta"] = df_tabla["precio_venta"].round(2)
    if "precio_competencia" in df_tabla.columns:
        df_tabla["precio_competencia"] = df_tabla["precio_competencia"].round(2)
    df_tabla["unidades_predichas"] = df_tabla["unidades_predichas"].round(0)
    df_tabla["ingresos_proyectados"] = df_tabla["ingresos_proyectados"].round(2)

    df_mostrar = df_tabla[
        [
            "fecha",
            "dia_semana_nombre",
            "precio_venta",
            "precio_competencia",
            "descuento_porcentaje",
            "unidades_predichas",
            "ingresos_proyectados",
        ]
    ].copy()

    df_mostrar["Black Friday"] = np.where(
        df_mostrar.index.isin(
            df_resultado[df_resultado.get("es_black_friday", 0) == 1].index
        ),
        "🔥",
        "",
    )

    return df_mostrar


def mostrar_tabla_detalle(df_tabla: pd.DataFrame) -> None:
    def resaltar_black_friday(row):
        color = "background-color: #ffe8e8;"
        return [color if "🔥" in str(v) else "" for v in row]

    st.dataframe(df_tabla.style.apply(resaltar_black_friday, axis=1))


def mostrar_kpis_principales(kpis: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Unidades totales", formatear_unidades(kpis.get("unidades_totales", 0))
        )
    with col2:
        st.metric(
            "Ingresos proyectados", formatear_euros(kpis.get("ingresos_totales", 0))
        )
    with col3:
        st.metric("Precio medio", formatear_euros(kpis.get("precio_promedio", 0)))
    with col4:
        st.metric("Descuento medio", f"{kpis.get('descuento_promedio', 0):.1f}%")


def mostrar_comparativa_escenarios(resumen_escenarios: dict) -> None:
    st.subheader("📊 Comparativa de escenarios de competencia")

    col_a, col_b, col_c = st.columns(3)

    for i, (nombre, kpis) in enumerate(resumen_escenarios.items()):
        with [col_a, col_b, col_c][i]:
            st.markdown(f"**{nombre}**")
            st.metric("Unidades", formatear_unidades(kpis.get("unidades_totales", 0)))
            st.metric("Ingresos", formatear_euros(kpis.get("ingresos_totales", 0)))


def main() -> None:
    st.set_page_config(
        page_title="Simulador de Ventas Noviembre 2025",
        page_icon="📈",
        layout="wide",
    )

    st.markdown(
        """<style>.main { background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%); }</style>""",
        unsafe_allow_html=True,
    )

    st.title("📈 Simulador de Ventas - Noviembre 2025")
    st.markdown(
        "Simula escenarios de precio y competencia para visualizar el impacto en las ventas."
    )

    ruta_base = Path(__file__).parent.parent
    ruta_csv = ruta_base / "data" / "processed" / "df_inferencia_transformado.csv"
    ruta_modelo = ruta_base / "modelo_pipeline.joblib"

    try:
        df = cargar_datos_inferencia(str(ruta_csv))
    except Exception as e:
        st.error(
            f"No se pudieron cargar los datos de inferencia. "
            f"Ejecuta primero: python main.py etl\n\nDetalle: {e}"
        )
        st.stop()

    try:
        inference = InferencePipeline(str(ruta_modelo))
    except Exception as e:
        st.error(
            f"No se pudo cargar el modelo. "
            f"Ejecuta primero: python main.py train\n\nDetalle: {e}"
        )
        st.stop()

    productos = sorted(df["nombre"].unique().tolist())

    with st.sidebar:
        st.markdown("## 🎛️ Controles de Simulación")

        producto_seleccionado = st.selectbox("Producto", productos)

        descuento = st.slider(
            "Ajuste de descuento (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
        )

        escenario_competencia = st.radio(
            "Escenario de competencia",
            ("Actual (0%)", "Competencia -5%", "Competencia +5%"),
        )

        if escenario_competencia == "Actual (0%)":
            ajuste_comp = 0.0
        elif escenario_competencia == "Competencia -5%":
            ajuste_comp = -5.0
        else:
            ajuste_comp = 5.0

        simular = st.button("🚀 Simular Ventas", use_container_width=True)

    df_producto = df[df["nombre"] == producto_seleccionado].copy()
    df_producto = df_producto.sort_values("fecha").reset_index(drop=True)

    st.markdown("---")
    st.header(f"🛒 Simulación para {producto_seleccionado} - Noviembre 2025")

    if not simular:
        st.info(
            "Ajusta los controles en la barra lateral y pulsa **'Simular Ventas'**."
        )
        return

    with st.spinner("Calculando predicciones..."):
        df_escenario_base, kpis_base = inference.predecir_producto(
            df_producto=df_producto,
            descuento_pct=float(descuento),
            ajuste_competencia_pct=float(ajuste_comp),
            recursive=True,
        )

        resultados = inference.predecir_multi_escenario(
            df_producto,
            descuento_pct=float(descuento),
            escenarios_competencia=[-5.0, 0.0, 5.0],
        )

    st.markdown("### 🔍 KPIs principales del escenario seleccionado")
    mostrar_kpis_principales(kpis_base)

    st.markdown("---")
    st.subheader("📉 Predicción diaria de unidades vendidas")
    crear_grafico_predicciones(df_escenario_base, producto_seleccionado)

    st.markdown("---")
    st.subheader("📋 Detalle diario de la simulación")
    df_tabla = preparar_tabla_detalle(df_escenario_base)
    mostrar_tabla_detalle(df_tabla)

    st.markdown("---")
    mostrar_comparativa_escenarios(resultados)


if __name__ == "__main__":
    main()
