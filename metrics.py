import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def metricas_regresion(y_real, y_pred):
    ae = np.abs(y_real - y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    y_real_safe = np.where(np.array(y_real) == 0, 1, np.array(y_real))
    mape = np.mean(np.abs((np.array(y_real) - np.array(y_pred)) / y_real_safe)) * 100

    return {
        "AE promedio": round(float(np.mean(ae)), 4),
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 4)
    }

def metricas_clasificacion(y_real, y_pred, etiquetas, clases):
    matriz = confusion_matrix(y_real, y_pred, labels=etiquetas)

    resumen = {
        "Accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "Precision": round(float(precision_score(y_real, y_pred, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_real, y_pred, average="weighted", zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_real, y_pred, average="weighted", zero_division=0)), 4)
    }

    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real: {c}" for c in clases],
        columns=[f"Predicho: {c}" for c in clases]
    )

    reporte = classification_report(
        y_real,
        y_pred,
        target_names=clases,
        zero_division=0,
        output_dict=True
    )

    return resumen, matriz_df, pd.DataFrame(reporte).transpose()
