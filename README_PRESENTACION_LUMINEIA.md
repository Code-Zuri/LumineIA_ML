# LumineIA + Machine Learning

## Guion para presentación

# 1. ¿Qué es LumineIA?

LumineIA es un asistente inteligente que aprende del comportamiento del
usuario para ofrecer respuestas y recomendaciones personalizadas.

A diferencia de un chatbot tradicional, LumineIA no responde únicamente
a la conversación actual. También analiza el historial del usuario y
aprende patrones de comportamiento utilizando Machine Learning.

Los datos analizados incluyen:

-   Uso del celular.
-   Horas estimadas de sueño.
-   Recordatorios creados y completados.
-   Tareas pendientes.
-   Conversaciones con Lumi.
-   Sentimiento de las conversaciones.
-   Productividad diaria.

El objetivo es comprender mejor al usuario para ofrecer una ayuda
personalizada.

------------------------------------------------------------------------

# 2. Flujo general del sistema

Usuario ↓ Base de datos ↓ Limpieza y preparación de datos ↓
Entrenamiento del modelo ↓ Predicción del estado del usuario ↓ Respuesta
personalizada de Lumi

El Machine Learning convierte datos históricos en conocimiento útil.

------------------------------------------------------------------------

# 3. Variables del modelo

## Variables independientes (Entradas)

Son los datos utilizados para entrenar el modelo.

Ejemplos:

-   Desbloqueos durante el día
-   Desbloqueos durante la noche
-   Tiempo de uso del celular
-   Tiempo sin usar el teléfono por la noche
-   Horas de sueño estimadas
-   Recordatorios creados
-   Recordatorios completados
-   Tareas pendientes
-   Número de conversaciones con Lumi
-   Sentimiento de la conversación
-   Productividad

## Variable dependiente (Salida)

Estado del usuario.

Puede ser:

-   Productivo
-   Normal
-   Cansado
-   Estresado
-   Desorganizado

------------------------------------------------------------------------

# 4. ¿Qué modelos usamos?

## Regresión Lineal

Objetivo:

Estimar las horas de sueño utilizando el tiempo sin uso nocturno.

Salida:

Valor numérico.

Ejemplo:

7.4 horas de sueño.

------------------------------------------------------------------------

## Regresión Múltiple

Objetivo:

Calcular un nivel estimado de productividad.

Utiliza varias variables al mismo tiempo.

Salida:

Un valor entre 0 y 100.

------------------------------------------------------------------------

## Clasificación

Objetivo:

Clasificar el estado actual del usuario.

Salida:

Una categoría.

Ejemplo:

"Cansado"

------------------------------------------------------------------------

# 5. ¿Por qué Random Forest?

Random Forest es el modelo principal del proyecto.

¿Por qué?

• Es muy preciso.

• Tolera ruido en los datos.

• Trabaja muy bien con muchas variables.

• Reduce el sobreajuste.

• Permite conocer cuáles variables influyen más.

Funcionamiento:

Árbol 1 → Productivo

Árbol 2 → Cansado

Árbol 3 → Productivo

Árbol 4 → Productivo

Resultado final:

Productivo

Cada árbol vota y gana la mayoría.

Por eso Random Forest es más robusto que un solo Árbol de Decisión.

------------------------------------------------------------------------

# 6. Árbol de Decisión

El árbol toma decisiones mediante preguntas.

Ejemplo:

¿Durmió menos de 6 horas?

↓

Sí

↓

¿Tiene muchas tareas pendientes?

↓

Sí

↓

Estado = Cansado

Cada nodo representa una decisión.

Cada hoja representa una clasificación final.

El árbol es fácil de explicar porque sus reglas son visibles.

------------------------------------------------------------------------

# 7. Red Neuronal

La red neuronal es el modelo más parecido al funcionamiento del cerebro
humano.

Está formada por neuronas artificiales organizadas en capas.

En este proyecto se utiliza un MLP (Multi Layer Perceptron).

Arquitectura:

Capa de entrada

↓

Capa oculta 1

↓

Capa oculta 2

↓

Capa de salida

## Capa de entrada

Recibe todas las variables del usuario.

Por ejemplo:

-   Horas de sueño
-   Tiempo de uso del celular
-   Recordatorios
-   Conversaciones
-   Productividad
-   Sentimiento
-   Tareas pendientes

Cada variable entra como un número.

------------------------------------------------------------------------

## Capas ocultas

Aquí ocurre el aprendizaje.

Cada neurona realiza tres pasos:

1.  Multiplica cada dato por un peso.

2.  Suma todos los resultados.

3.  Aplica una función de activación (ReLU).

Si el patrón es importante, la neurona se activa.

Si no, prácticamente se ignora.

Cada capa descubre relaciones más complejas.

Ejemplo:

Una neurona puede aprender:

"Poco sueño + muchas tareas + sentimiento negativo"

Otra aprende:

"Muchos recordatorios completados + buen descanso"

Otra aprende:

"Uso excesivo del celular durante la noche"

Ninguna de esas reglas fue programada manualmente.

La red las descubre sola durante el entrenamiento.

------------------------------------------------------------------------

## Entrenamiento

Durante el entrenamiento ocurre este ciclo:

1.  La red recibe un usuario.

2.  Predice un estado.

3.  Compara la predicción con el estado real.

4.  Calcula el error (Loss).

5.  Ajusta automáticamente los pesos.

6.  Repite cientos de veces.

Con cada iteración el error disminuye.

Cuando termina, la red reconoce patrones nuevos.

------------------------------------------------------------------------

## ¿Qué es el Loss?

El Loss representa el error del modelo.

Loss alto:

La red está aprendiendo.

Loss bajo:

La red ya reconoce correctamente los patrones.

Nuestro dashboard muestra el Loss final y las iteraciones realizadas.

------------------------------------------------------------------------

## ¿Por qué no usamos solamente una Red Neuronal?

Porque Random Forest ofrece varias ventajas para este proyecto:

-   Se puede explicar fácilmente.
-   Permite saber qué variables influyen más.
-   Es más estable con pocos datos.
-   Requiere menos entrenamiento.

La Red Neuronal será más útil cuando LumineIA tenga miles de usuarios y
millones de conversaciones.

------------------------------------------------------------------------

# 8. Métricas

Regresión:

AE MAE MSE RMSE R² MAPE

Clasificación:

Accuracy Precision Recall F1 Score Matriz de Confusión

Estas métricas indican qué tan bueno es el modelo.

------------------------------------------------------------------------

# 9. Demostración

Durante la presentación se modifica un usuario.

El modelo analiza sus datos.

Predice un estado.

Lumi genera una recomendación personalizada.

Así se demuestra que el Machine Learning no solo almacena información,
sino que aprende del comportamiento del usuario.

------------------------------------------------------------------------

# 10. Conclusión

LumineIA transforma datos en decisiones inteligentes.

El proyecto utiliza Machine Learning para aprender hábitos, detectar
patrones y adaptar automáticamente las respuestas del asistente.

Random Forest es el modelo principal por su precisión e
interpretabilidad.

La Red Neuronal representa la evolución futura del proyecto, cuando
exista una base de datos mucho mayor con conversaciones reales.
