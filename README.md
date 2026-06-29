# LumineIA ML Dashboard


## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar dashboard

```bash
streamlit run app.py
```

## Ver resumen en terminal

```bash
python terminal_summary.py
```

## Cómo explicar la red neuronal

La red neuronal toma las variables del usuario como entrada:

- desbloqueos del celular
- uso nocturno
- horas de sueño
- recordatorios
- tareas pendientes
- mensajes con Lumi
- sentimiento de conversación
- productividad

Después normaliza los datos, los pasa por capas ocultas y aprende pesos internos.
El entrenamiento busca reducir el `loss`, que representa el error del modelo.
Cuando termina, la red puede clasificar al usuario como Cansado, Estresado,
Productivo, Desorganizado o Normal.

En este proyecto, la red neuronal se usa como comparación avanzada.
El modelo principal recomendado sigue siendo Random Forest porque es más fácil
de explicar frente a un público y permite mostrar qué variables influyen más.


