from data_generator import generar_datos_lumineia
from models import entrenar_clasificador_estado, regresion_lineal_sueno, regresion_multiple_productividad, info_red_neuronal

def main():
    df = generar_datos_lumineia(n=800, seed=42)

    print("\n==============================")
    print("RESUMEN LUMINEIA MACHINE LEARNING")
    print("==============================")
    print(f"Registros usados: {len(df)}")
    print(f"Sueño promedio: {df['horas_sueno_estimadas'].mean():.2f} horas")
    print(f"Productividad promedio: {df['productividad_score'].mean():.2f}")
    print(f"Uso celular promedio: {df['tiempo_uso_celular_min'].mean():.2f} minutos")
    print("\nDistribución de estados:")
    print(df["estado_usuario"].value_counts())

    _, metricas_rl, _, _, _ = regresion_lineal_sueno(df)
    _, metricas_rm, _, _, _, _ = regresion_multiple_productividad(df)

    print("\nREGRESIÓN LINEAL - horas de sueño")
    print(metricas_rl)

    print("\nREGRESIÓN MÚLTIPLE - productividad")
    print(metricas_rm)

    for algoritmo in ["Árbol de Decisión", "Random Forest", "Red Neuronal"]:
        modelo, encoder, resumen, _, _, _, _, _, _, _ = entrenar_clasificador_estado(
            df,
            algoritmo=algoritmo
        )
        print(f"\nCLASIFICACIÓN - {algoritmo}")
        print(resumen)

        if algoritmo == "Red Neuronal":
            print("\nDetalle red neuronal:")
            print(info_red_neuronal(modelo))

    print("\nConclusión:")
    print("Random Forest es el modelo principal recomendado por estabilidad e interpretación.")
    print("La red neuronal aprende patrones no lineales, pero requiere más datos reales para justificar mejor su uso.")
    print("==============================\n")

if __name__ == "__main__":
    main()
