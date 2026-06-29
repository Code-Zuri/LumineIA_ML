import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from metrics import metricas_regresion, metricas_clasificacion

FEATURES_ESTADO = [
    "desbloqueos_dia",
    "desbloqueos_noche",
    "tiempo_uso_celular_min",
    "tiempo_sin_uso_nocturno_min",
    "horas_sueno_estimadas",
    "recordatorios_creados",
    "recordatorios_completados",
    "tareas_pendientes",
    "mensajes_con_lumi",
    "sentimiento_score",
    "productividad_score",
    "riesgo_desorganizacion"
]

def regresion_lineal_sueno(df):
    X = df[["tiempo_sin_uso_nocturno_min"]]
    y = df["horas_sueno_estimadas"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    return modelo, metricas_regresion(y_test.values, pred), X_test, y_test, pred

def regresion_multiple_productividad(df):
    X = df[[
        "horas_sueno_estimadas",
        "recordatorios_completados",
        "tareas_pendientes",
        "tiempo_uso_celular_min",
        "mensajes_con_lumi",
        "sentimiento_score",
        "riesgo_desorganizacion"
    ]]
    y = df["productividad_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    coeficientes = pd.DataFrame({
        "Variable": X.columns,
        "Coeficiente": modelo.coef_
    }).sort_values("Coeficiente", ascending=False)

    return modelo, metricas_regresion(y_test.values, pred), coeficientes, X_test, y_test, pred

def preparar_clasificacion(df):
    X = df[FEATURES_ESTADO]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["estado_usuario"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoder

def crear_modelo(algoritmo, criterio="gini", svm_kernel="rbf"):
    if algoritmo == "Árbol de Decisión":
        return DecisionTreeClassifier(criterion=criterio, max_depth=5, random_state=42)

    if algoritmo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250,
            criterion=criterio,
            max_depth=9,
            random_state=42
        )

    if algoritmo == "Regresión Logística":
        return Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=2000))
        ])

    if algoritmo == "SVM":
        return Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", SVC(kernel=svm_kernel))
        ])

    if algoritmo == "Red Neuronal":
        return Pipeline([
            ("escalado", StandardScaler()),
            ("modelo", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42
            ))
        ])

    return RandomForestClassifier(random_state=42)

def entrenar_clasificador_estado(df, algoritmo="Random Forest", criterio="gini", svm_kernel="rbf"):
    X_train, X_test, y_train, y_test, encoder = preparar_clasificacion(df)
    modelo = crear_modelo(algoritmo, criterio, svm_kernel)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    etiquetas = list(range(len(encoder.classes_)))
    clases = list(encoder.classes_)
    resumen, matriz, reporte = metricas_clasificacion(y_test, pred, etiquetas, clases)

    return modelo, encoder, resumen, matriz, reporte, X_train, X_test, y_train, y_test, pred

def importancia_modelo(modelo):
    if hasattr(modelo, "feature_importances_"):
        return pd.DataFrame({
            "Variable": FEATURES_ESTADO,
            "Importancia": modelo.feature_importances_
        }).sort_values("Importancia", ascending=True)

    if hasattr(modelo, "named_steps"):
        interno = modelo.named_steps.get("modelo")
        if hasattr(interno, "coefs_"):
            return None

    return None

def reglas_arbol(modelo):
    if isinstance(modelo, DecisionTreeClassifier):
        return export_text(modelo, feature_names=FEATURES_ESTADO)
    return ""

def info_red_neuronal(modelo):
    if not hasattr(modelo, "named_steps"):
        return None

    interno = modelo.named_steps.get("modelo")
    if not hasattr(interno, "coefs_"):
        return None

    capas = [interno.coefs_[0].shape[0]]
    for pesos in interno.coefs_:
        capas.append(pesos.shape[1])

    return {
        "capas": capas,
        "iteraciones": interno.n_iter_,
        "loss_final": round(float(interno.loss_), 6),
        "funcion_activacion": interno.activation,
        "optimizador": interno.solver
    }

def predecir_estado_manual(modelo, encoder, valores):
    df_usuario = pd.DataFrame([valores], columns=FEATURES_ESTADO)
    pred = modelo.predict(df_usuario)[0]
    estado = encoder.inverse_transform([pred])[0]

    recomendaciones = {
        "Cansado": "Lumi sugiere descanso, disminuir uso nocturno y organizar el día con calma.",
        "Estresado": "Lumi divide tareas, prioriza pendientes y responde con apoyo emocional.",
        "Productivo": "Lumi refuerza el progreso y mantiene la rutina actual.",
        "Desorganizado": "Lumi crea un plan de prioridades y recordatorios.",
        "Normal": "Lumi da seguimiento personalizado normal."
    }

    return estado, recomendaciones.get(estado, "Lumi personaliza la respuesta con base en el contexto.")
