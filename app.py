import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"

from data_generator import generar_datos_lumineia
from models import (
    regresion_lineal_sueno,
    regresion_multiple_productividad,
    entrenar_clasificador_estado,
    importancia_modelo,
    reglas_arbol,
    info_red_neuronal,
    predecir_estado_manual,
    FEATURES_ESTADO
)


def aplicar_estilo_figura(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=18, color="#ffffff"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)"
        )
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


st.set_page_config(
    page_title="LumineIA ML Dashboard",
    page_icon="🧠",
    layout="wide"
)


st.markdown("""
<style>
/* ====== LUMINEIA DARK MINIMAL UI ====== */

.stApp {
    background: linear-gradient(135deg, #09090f 0%, #10111a 45%, #141625 100%);
    color: #f4f4f5;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(12, 13, 22, 0.96);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Titles */
h1 {
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em;
    color: #ffffff;
    margin-bottom: 0.25rem;
}

h2 {
    font-size: 1.45rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-top: 1.5rem;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #f5f5f5;
}

/* Caption and text */
p, li, span {
    color: #d4d4d8;
}

[data-testid="stCaptionContainer"] {
    color: #a1a1aa;
}

/* Cards */
div[data-testid="stMetric"],
.stAlert,
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 14px 40px rgba(0,0,0,0.22);
}

/* Metrics */
div[data-testid="stMetric"] {
    min-height: 105px;
}

[data-testid="stMetricLabel"] {
    color: #a1a1aa !important;
    font-size: 0.82rem !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    background: #7c3aed;
    color: white;
    font-weight: 700;
    padding: 0.65rem 1.15rem;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(255,255,255,0.28);
    filter: brightness(1.08);
}

/* Inputs */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input,
.stFileUploader {
    border-radius: 14px !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.09);
}

/* Plot containers */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.5rem;
    box-shadow: 0 14px 40px rgba(0,0,0,0.18);
}

/* Dividers */
hr {
    border-color: rgba(255,255,255,0.08);
    margin: 2rem 0;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Hero */
.hero-card {
    background: radial-gradient(circle at top left, rgba(124,58,237,0.28), transparent 35%),
                radial-gradient(circle at bottom right, rgba(37,99,235,0.25), transparent 35%),
                rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 26px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 18px 55px rgba(0,0,0,0.26);
}

.hero-title {
    font-size: 1.05rem;
    color: #e4e4e7;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 0.92rem;
    color: #a1a1aa;
}

.clean-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1rem 1.15rem;
    height: 100%;
}

.clean-card strong {
    color: #ffffff;
}

.small-muted {
    color: #a1a1aa;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <br>
<div class="hero-card">
    <div class="hero-title">LumineIA | Modelo de Machine Learning</div>         
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuración")
    archivo = st.file_uploader("Cargar CSV", type=["csv"])
    cantidad = st.slider("Registros sintéticos", 400, 3000, 800)
    seed = st.number_input("Seed", min_value=1, value=42)

if archivo:
    df = pd.read_csv(archivo)
    st.success("Base de datos cargada correctamente.")
else:
    df = generar_datos_lumineia(cantidad, seed)

columnas_requeridas = [
    "desbloqueos_dia", "desbloqueos_noche", "tiempo_uso_celular_min",
    "tiempo_sin_uso_nocturno_min", "horas_sueno_estimadas",
    "recordatorios_creados", "recordatorios_completados", "tareas_pendientes",
    "mensajes_con_lumi", "sentimiento_score", "productividad_score",
    "riesgo_desorganizacion", "estado_usuario", "descanso_bueno"
]
faltantes = [c for c in columnas_requeridas if c not in df.columns]
if faltantes:
    st.error("El CSV no tiene todas las columnas necesarias.")
    st.write(faltantes)
    st.stop()

# Resumen también en terminal
print("\n===== RESUMEN LUMINEIA ML =====")
print(f"Registros: {len(df)}")
print(f"Sueño promedio: {df['horas_sueno_estimadas'].mean():.2f}")
print(f"Productividad promedio: {df['productividad_score'].mean():.2f}")
print("Estados:")
print(df["estado_usuario"].value_counts())
print("================================\n")

st.header("1. Resumen")

a, b, c, d = st.columns(4)
a.metric("Registros", len(df))
b.metric("Sueño promedio", round(df["horas_sueno_estimadas"].mean(), 2))
c.metric("Productividad promedio", round(df["productividad_score"].mean(), 2))
d.metric("Uso celular promedio", round(df["tiempo_uso_celular_min"].mean(), 2))

st.info(
    "Objetivo: predecir el estado del usuario para que Lumi responda de forma personalizada, "
    "usando hábitos, tareas, descanso y conversaciones."
)

with st.expander("Ver muestra de la base de datos"):
    st.dataframe(df.head(150), use_container_width=True)
    st.download_button(
        "Descargar CSV generado",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lumineia_datos.csv",
        mime="text/csv"
    )

st.divider()

st.header("2. Análisis")

g1, g2 = st.columns(2)

with g1:
    fig = px.histogram(
        df,
        x="estado_usuario",
        title="Distribución de estados del usuario",
        labels={"estado_usuario": "Estado del usuario"}
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

with g2:
    fig = px.scatter(
        df,
        x="tiempo_uso_celular_min",
        y="horas_sueno_estimadas",
        color="estado_usuario",
        title="Uso del celular vs horas de sueño",
        labels={
            "tiempo_uso_celular_min": "Uso celular (min)",
            "horas_sueno_estimadas": "Horas de sueño"
        }
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

g3, g4 = st.columns(2)

with g3:
    fig = px.scatter(
        df,
        x="tareas_pendientes",
        y="productividad_score",
        color="estado_usuario",
        title="Tareas pendientes vs productividad",
        labels={
            "tareas_pendientes": "Tareas pendientes",
            "productividad_score": "Productividad"
        }
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

with g4:
    promedio = df.groupby("estado_usuario", as_index=False)["desbloqueos_noche"].mean()
    fig = px.bar(
        promedio,
        x="estado_usuario",
        y="desbloqueos_noche",
        title="Promedio de desbloqueos nocturnos por estado",
        labels={
            "estado_usuario": "Estado",
            "desbloqueos_noche": "Desbloqueos nocturnos promedio"
        }
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

st.caption("Las gráficas son interactivas: puedes hacer zoom, mover, seleccionar y descargar como imagen.")

st.divider()

st.header("3. Conceptos ")

conceptos = pd.DataFrame({
    "Elemento": [
        "Entrada",
        "Modelo",
        "Salida",
        "Árbol de Decisión",
        "Random Forest",
        "Red Neuronal"
    ],
    "Explicación corta": [
        "Datos del usuario: uso, tareas, sueño y conversación.",
        "Algoritmo que aprende patrones.",
        "Estado del usuario y recomendación de Lumi.",
        "Reglas claras tipo: si pasa esto, entonces...",
        "Muchos árboles votan la mejor respuesta.",
        "Capas de neuronas aprenden relaciones más complejas."
    ]
})
st.dataframe(conceptos, hide_index=True, use_container_width=True)

st.divider()

st.header("4. Modelo principal: clasificación del estado del usuario")

col1, col2, col3 = st.columns(3)
with col1:
    algoritmo = st.selectbox(
        "Modelo",
        ["Random Forest", "Árbol de Decisión", "Red Neuronal", "Regresión Logística", "SVM"]
    )
with col2:
    criterio = st.selectbox("Criterio de división", ["gini", "entropy"])
with col3:
    svm_kernel = st.selectbox("Tipo de SVM", ["linear", "rbf", "poly"])

modelo, encoder, resumen, matriz, reporte, X_train, X_test, y_train, y_test, pred = entrenar_clasificador_estado(
    df,
    algoritmo=algoritmo,
    criterio=criterio,
    svm_kernel=svm_kernel
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", resumen["Accuracy"])
m2.metric("Precision", resumen["Precision"])
m3.metric("Recall", resumen["Recall"])
m4.metric("F1 Score", resumen["F1 Score"])

st.subheader("Matriz de confusión")
st.dataframe(matriz, use_container_width=True)

st.subheader("Reporte por clase")
st.dataframe(reporte, use_container_width=True)

importancia = importancia_modelo(modelo)
if importancia is not None:
    st.subheader("Importancia de variables")
    fig = px.bar(
        importancia,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Qué variables influyen más en la predicción"
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

if algoritmo == "Árbol de Decisión":
    with st.expander("Reglas reales aprendidas por el árbol"):
        st.code(reglas_arbol(modelo))

st.divider()

st.header("5. Red neuronal: cómo se aplica y cómo funciona")

st.write(
    "En LumineIA, la red neuronal recibe las mismas variables del usuario, "
    "las transforma en valores numéricos normalizados y aprende patrones para clasificar el estado."
)

rn_modelo, rn_encoder, rn_resumen, rn_matriz, rn_reporte, *_ = entrenar_clasificador_estado(
    df,
    algoritmo="Red Neuronal"
)
rn_info = info_red_neuronal(rn_modelo)

n1, n2, n3 = st.columns(3)
n1.metric("Capas", " → ".join(map(str, rn_info["capas"])))
n2.metric("Iteraciones", rn_info["iteraciones"])
n3.metric("Loss final", rn_info["loss_final"])

st.markdown("""
**Lectura sencilla para exposición:**

1. Las variables entran a la red neuronal.  
2. La red usa capas ocultas para encontrar patrones.  
3. Ajusta sus pesos internos durante el entrenamiento.  
4. El error o `loss` baja conforme aprende.  
5. Al final predice el estado del usuario.
""")

st.subheader("Rendimiento de la red neuronal")
rn1, rn2, rn3, rn4 = st.columns(4)
rn1.metric("Accuracy", rn_resumen["Accuracy"])
rn2.metric("Precision", rn_resumen["Precision"])
rn3.metric("Recall", rn_resumen["Recall"])
rn4.metric("F1 Score", rn_resumen["F1 Score"])

st.dataframe(rn_matriz, use_container_width=True)

st.divider()

st.header("6. Regresión: sueño y productividad")

r1, r2 = st.columns(2)

with r1:
    _, met_rl, X_rl, y_rl, pred_rl = regresion_lineal_sueno(df)
    st.subheader("Regresión lineal")
    st.caption("Predice horas de sueño con tiempo sin uso nocturno.")
    st.json(met_rl)

    plot_rl = pd.DataFrame({
        "tiempo_sin_uso_nocturno_min": X_rl["tiempo_sin_uso_nocturno_min"],
        "real": y_rl,
        "prediccion": pred_rl
    })
    fig = px.scatter(
        plot_rl,
        x="tiempo_sin_uso_nocturno_min",
        y=["real", "prediccion"],
        title="Sueño real vs predicción"
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

with r2:
    _, met_rm, coef, X_rm, y_rm, pred_rm = regresion_multiple_productividad(df)
    st.subheader("Regresión múltiple")
    st.caption("Predice productividad usando varias variables.")
    st.json(met_rm)

    plot_rm = pd.DataFrame({
        "Productividad real": y_rm,
        "Productividad predicha": pred_rm
    })
    fig = px.scatter(
        plot_rm,
        x="Productividad real",
        y="Productividad predicha",
        title="Productividad real vs predicha"
    )
    st.plotly_chart(aplicar_estilo_figura(fig), use_container_width=True)

st.divider()

st.header("7. Ejemplo práctico: predicción real para un usuario")

st.write("Este bloque demuestra la función real del modelo: convertir datos en una respuesta personalizada de Lumi.")

c1, c2, c3 = st.columns(3)

with c1:
    desbloqueos_dia = st.number_input("Desbloqueos día", 0, 300, 90)
    desbloqueos_noche = st.number_input("Desbloqueos noche", 0, 100, 20)
    tiempo_uso_celular_min = st.number_input("Uso celular en minutos", 0, 1000, 360)
    tiempo_sin_uso_nocturno_min = st.number_input("Tiempo sin uso nocturno", 0, 900, 360)

with c2:
    horas_sueno_estimadas = st.number_input("Horas de sueño", 0.0, 12.0, 5.8)
    recordatorios_creados = st.number_input("Recordatorios creados", 0, 30, 5)
    recordatorios_completados = st.number_input("Recordatorios completados", 0, 30, 2)
    tareas_pendientes = st.number_input("Tareas pendientes", 0, 30, 9)

with c3:
    mensajes_con_lumi = st.number_input("Mensajes con Lumi", 0, 200, 25)
    sentimiento_score = st.selectbox("Sentimiento", [-1, 0, 1], index=0)
    productividad_score = st.number_input("Productividad", 0.0, 100.0, 45.0)
    riesgo_desorganizacion = st.number_input("Riesgo de desorganización", 0.0, 100.0, 65.0)

valores_usuario = [
    desbloqueos_dia,
    desbloqueos_noche,
    tiempo_uso_celular_min,
    tiempo_sin_uso_nocturno_min,
    horas_sueno_estimadas,
    recordatorios_creados,
    recordatorios_completados,
    tareas_pendientes,
    mensajes_con_lumi,
    sentimiento_score,
    productividad_score,
    riesgo_desorganizacion
]

if st.button("Predecir estado"):
    estado, recomendacion = predecir_estado_manual(modelo, encoder, valores_usuario)
    st.success(f"Estado predicho: {estado}")
    st.info(f"Respuesta de Lumi: {recomendacion}")

