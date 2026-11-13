# Enfermedad cardiaca con indicadores de salud: informe CRISP-DM

## 1. Comprensión del negocio
Identificar qué factores de salud y hábitos se asocian con infarto o enfermedad cardiaca (`HeartDiseaseorAttack`) y estimar la probabilidad de que una persona la presente, para apoyar la prevención. Los datos son autoreportados y de corte transversal: sirven para detectar asociaciones, no causas.

## 2. Comprensión de los datos
- Fuente: [Heart Disease Health Indicators (alexteboul)](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset), CC0, basado en la encuesta BRFSS 2015 (CDC).
- 253.680 filas y 22 columnas. Hay 23.899 duplicados; sin ellos quedan **229.781 registros**.
- Sin nulos.
- Prevalencia del evento: 10,3 % sin duplicados. El dataset está desbalanceado, por eso se usan métricas como PR-AUC y recall.
- Fuertes asociaciones con el evento (p < 0,001): edad, salud general, dificultad para caminar, accidente cerebrovascular, presión alta, colesterol alto, diabetes, sexo y nivel de ingreso.

## 3. Preparación de los datos
- Validación de códigos BRFSS y rangos: sin errores.
- Eliminación de duplicados.
- Variables derivadas: edad aproximada (en años), categoría de IMC, número de condiciones (presión, colesterol, diabetes, tabaco, ACV) e inactividad física.
- Carga en SQLite: `data/heart_disease.db`, tablas `raw_encuesta` (originales) y `encuestados` (limpios).

## 4. Modelamiento
Se compararon una línea base (clase mayoritaria), regresión logística con `class_weight="balanced"` y gradient boosting con pesos de clase. Validación cruzada de 5 folds sobre el 80 % de entrenamiento.

## 5. Evaluación

**Validación cruzada (5 folds, entrenamiento):**

| Modelo | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Línea base | 0,500 | 0,103 | 0,000 |
| Regresión logística | 0,836 ± 0,003 | 0,373 | 0,383 |
| Gradient boosting | 0,838 ± 0,003 | 0,378 | 0,379 |

**Conjunto de prueba (20 %, 45.957 registros):**

| Modelo | ROC-AUC | PR-AUC | Precisión (0,5) | Recall (0,5) | F1 (0,5) |
|---|---|---|---|---|---|
| Línea base | 0,500 | 0,103 | 0,000 | 0,000 | 0,000 |
| Regresión logística | 0,836 | 0,373 | 0,252 | 0,778 | 0,381 |
| Gradient boosting | **0,838** | **0,377** | 0,246 | **0,801** | 0,376 |

**Lectura de los resultados:**
- Los dos modelos dan resultados casi iguales. La regresión logística es más simple y fácil de explicar, así que es la recomendada.
- Con umbral 0,5, el modelo detecta cerca del 78 % de los casos reales, pero una parte de las alertas son falsas (precisión ≈ 25 %). Para un tamizaje, esto suele ser aceptable. El umbral se puede ajustar según el costo de un falso negativo frente a un falso positivo.

**Variables más influyentes** (coeficientes de la regresión logística y permutación del gradient boosting):
1. Edad y salud general (los dos métodos coinciden).
2. Accidente cerebrovascular previo.
3. Sexo, colesterol alto y presión alta.
4. Diabetes y tabaquismo tienen efecto menor.

## 6. Presentación de resultados
- Figuras en `reports/figures/`: distribuciones, correlación, tasas por edad y salud, curvas ROC y matriz de confusión.
- Tablas en `reports/`: asociaciones, métricas y coeficientes.

**Limitaciones:**
- Los datos son autoreportados y de corte transversal; no permiten inferir causalidad.
- Se usó el dataset sin duplicados. Si la encuesta tiene personas repetidas por diseño, eliminarlas puede cambiar algo las métricas.
- No se hizo ajuste de hiperparámetros; los resultados pueden mejorar con búsqueda de hiperparámetros.

## Consideraciones éticas

- **Datos de salud:** el dataset proviene de una encuesta de los CDC, sin nombres ni identificadores directos. Los resultados se presentan siempre en agregado y no deben usarse para identificar personas.
- **Uso del modelo:** las predicciones sirven para orientar la prevención y la conversación clínica. No reemplazan el diagnóstico médico ni deben usarse para decidir seguros, empleo o acceso a servicios.
- **Sesgo y equidad:** la encuesta es telefónica y autoreportada, por lo que puede no representar bien a todos los grupos. El desempeño por sexo y edad se revisó (sección 5.6), y muestra diferencias de recall y precisión. Hay que revisar esas brechas antes de cualquier uso real.
- **Falsos negativos:** un modelo con precisión baja y recall alto genera muchas alertas falsas. Esa carga debe considerarse frente a los costos de pasar por alto un caso.
- **Licencia:** el dataset se publica bajo CC0 (dominio público), por lo que su uso es libre con la atribución de la fuente.
