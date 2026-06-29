import numpy as np
import pandas as pd

def generar_datos_lumineia(n=800, seed=42):
    np.random.seed(seed)

    fechas = pd.date_range("2026-01-01", periods=n, freq="D")

    desbloqueos_dia = np.random.randint(25, 190, n)
    desbloqueos_noche = np.random.randint(0, 50, n)
    tiempo_uso_celular_min = np.random.randint(70, 720, n)
    tiempo_sin_uso_nocturno_min = np.random.randint(150, 650, n)

    recordatorios_creados = np.random.randint(0, 14, n)
    recordatorios_completados = np.array([
        np.random.randint(0, max(1, rc + 1)) for rc in recordatorios_creados
    ])

    tareas_pendientes = np.random.randint(0, 18, n)
    mensajes_con_lumi = np.random.randint(0, 90, n)

    # -1 = negativo, 0 = neutral, 1 = positivo
    sentimiento_score = np.random.choice([-1, 0, 1], size=n, p=[0.30, 0.35, 0.35])

    horas_sueno_estimadas = (
        tiempo_sin_uso_nocturno_min / 60
        - desbloqueos_noche * 0.045
        + np.random.normal(0, 0.40, n)
    )
    horas_sueno_estimadas = np.clip(horas_sueno_estimadas, 2.5, 10)

    productividad_score = (
        38
        + recordatorios_completados * 5.2
        - tareas_pendientes * 2.4
        + horas_sueno_estimadas * 4.1
        - tiempo_uso_celular_min * 0.035
        + sentimiento_score * 6.5
        + np.random.normal(0, 6, n)
    )
    productividad_score = np.clip(productividad_score, 0, 100)

    descanso_bueno = np.where(horas_sueno_estimadas >= 7, 1, 0)

    estado_usuario = []
    riesgo_desorganizacion = []
    recomendacion_lumi = []

    for i in range(n):
        riesgo = (
            tareas_pendientes[i] * 4
            + desbloqueos_noche[i] * 1.5
            + tiempo_uso_celular_min[i] * 0.03
            - recordatorios_completados[i] * 3
            - horas_sueno_estimadas[i] * 2
        )
        riesgo_desorganizacion.append(round(float(np.clip(riesgo, 0, 100)), 2))

        if horas_sueno_estimadas[i] < 6 and desbloqueos_noche[i] > 18:
            estado_usuario.append("Cansado")
            recomendacion_lumi.append("Reducir uso nocturno y sugerir descanso")
        elif sentimiento_score[i] == -1 and tareas_pendientes[i] >= 8:
            estado_usuario.append("Estresado")
            recomendacion_lumi.append("Organizar tareas y responder con apoyo")
        elif productividad_score[i] >= 72 and recordatorios_completados[i] >= 4:
            estado_usuario.append("Productivo")
            recomendacion_lumi.append("Mantener rutina y reforzar avance")
        elif tareas_pendientes[i] >= 11:
            estado_usuario.append("Desorganizado")
            recomendacion_lumi.append("Crear plan por prioridades")
        else:
            estado_usuario.append("Normal")
            recomendacion_lumi.append("Seguimiento personalizado normal")

    return pd.DataFrame({
        "fecha": fechas,
        "desbloqueos_dia": desbloqueos_dia,
        "desbloqueos_noche": desbloqueos_noche,
        "tiempo_uso_celular_min": tiempo_uso_celular_min,
        "tiempo_sin_uso_nocturno_min": tiempo_sin_uso_nocturno_min,
        "horas_sueno_estimadas": np.round(horas_sueno_estimadas, 2),
        "recordatorios_creados": recordatorios_creados,
        "recordatorios_completados": recordatorios_completados,
        "tareas_pendientes": tareas_pendientes,
        "mensajes_con_lumi": mensajes_con_lumi,
        "sentimiento_score": sentimiento_score,
        "productividad_score": np.round(productividad_score, 2),
        "riesgo_desorganizacion": riesgo_desorganizacion,
        "descanso_bueno": descanso_bueno,
        "estado_usuario": estado_usuario,
        "recomendacion_lumi": recomendacion_lumi
    })
