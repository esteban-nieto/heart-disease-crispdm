[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/esteban-nieto/heart-disease-crispdm/blob/main/notebooks/analisis_crispdm.ipynb)

# Enfermedad cardiaca: factores de riesgo y modelos predictivos (CRISP-DM)

Proyecto de ciencia de datos que estudia qué factores de salud y hábitos se asocian con infarto o enfermedad cardiaca, y qué tan bien se puede estimar ese riesgo con modelos predictivos. Sigue la metodología CRISP-DM en sus seis fases.

**Autor:** Alejandro Pedreros
**Fecha:** 13/11/2026

---

## Resumen de resultados

| Modelo | ROC-AUC (prueba) | PR-AUC (prueba) | Recall (umbral 0,5) | Precisión (umbral 0,5) |
|---|---|---|---|---|
| Línea base (mayoritaria) | 0,500 | 0,103 | 0,000 | 0,000 |
| Regresión logística | 0,836 | 0,373 | 0,778 | 0,252 |
| Random Forest | 0,837 | 0,371 | 0,794 | 0,249 |
| Gradient boosting | 0,838 | 0,377 | 0,801 | 0,246 |

- Los factores más asociados son **edad**, **salud general**, **sexo**, **colesterol alto** y **presión alta**.
- La regresión logística y el Random Forest rinden casi igual. Los intervalos de confianza por bootstrap se traslapan.
- El modelo ordena bien el riesgo, pero **no rinde igual por sexo ni por edad**: detecta más casos en mujeres y en mayores de 65, con menor precisión.
- Las probabilidades **no están calibradas**. Sirven para ordenar riesgo, no como probabilidad absoluta de infarto.

Los detalles están en el [informe técnico](reports/informe_tecnico.md) y en el [resumen ejecutivo](reports/resultados_crispdm.md).

---

## Datos

- **Fuente:** [Heart Disease Health Indicators (alexteboul)](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) en Kaggle, licencia CC0, basado en la encuesta BRFSS 2015 de los Centros para el Control y la Prevención de Enfermedades (CDC) de Estados Unidos.
- **Tamaño:** 253.680 registros originales, 22 columnas. Se eliminan 23.899 duplicados exactos y quedan **229.781 registros**.
- **Objetivo:** `HeartDiseaseorAttack` (infarto o enfermedad cardiaca), con 10,3 % de casos positivos.
- **Limitación de origen:** datos autoreportados y de corte transversal. Muestran asociaciones, no causas.

El archivo `data/heart_disease_health_indicators_BRFSS2015.csv` es el dato original sin cambios.

---

## Estructura del proyecto

```
heart-disease-crispdm/
├── data/
│   └── heart_disease_health_indicators_BRFSS2015.csv   Dato original (22,7 MB)
├── notebooks/
│   └── analisis_crispdm.ipynb                          Análisis completo: fases 1 a 4
├── reports/
│   ├── informe_tecnico.md / .docx                      Informe técnico paso a paso
│   ├── resultados_crispdm.md / .docx                   Resumen ejecutivo
│   ├── presentacion_resultados.pdf                     Presentación (15 páginas)
│   ├── presentacion_resultados.pptx                    Diapositivas
│   ├── *.csv                                           Tablas de métricas y análisis
│   └── figures/                                        Figuras 01 a 09
├── .gitignore
├── requirements.txt
└── README.md
```

El notebook crea `data/heart_disease.db` a partir del CSV original en la fase 2. Esa base no está en el repositorio.

---

## Cómo reproducir el análisis

### Opción 1: Google Colab (recomendada)

1. Haz clic en el botón **Abrir en Colab** de arriba.
2. Ejecuta la primera celda de configuración. Clona el repositorio en `/content/heart-disease-crispdm`, incluido el CSV, e instala las dependencias.
3. Ejecuta el resto de las celdas en orden (**Entorno de ejecución → Ejecutar todo**).

La fase 3 tarda unos 30 minutos en Colab, sobre todo por SHAP.

### Opción 2: local

Se recomienda Python 3.11 o 3.12. Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analisis_crispdm.ipynb
```

Ejecuta las celdas en orden. La celda de configuración detecta si estás en Colab o en local y crea las carpetas `data/` y `reports/figures/` si no existen.

Notas:
- Las versiones de las librerías están fijadas en `requirements.txt` para que los resultados coincidan.
- SHAP se calcula sobre una muestra de 300 registros de prueba, por costo computacional.

---

## Metodología (CRISP-DM)

1. **Comprensión del negocio:** qué factores se asocian con el evento y qué tan bien se puede estimar el riesgo para apoyar la prevención.
2. **Comprensión de los datos:** perfil de nulos, duplicados y rangos; pruebas de Mann-Whitney U y chi-cuadrado; correlaciones con el objetivo.
3. **Preparación:** validación de códigos contra el libro de códigos BRFSS, eliminación de duplicados, variables derivadas (edad aproximada, categoría de IMC, número de condiciones) y carga en SQLite. Las binarias quedan en 0/1 y las ordinales como enteros, para respetar su orden. La normalización (`StandardScaler`) solo se aplica a la regresión logística y se ajusta con el entrenamiento.
4. **Modelamiento:** línea base, regresión logística, Random Forest y gradient boosting. Partición estratificada 80/20 con semilla 42 y validación cruzada de 5 folds sobre el entrenamiento.
5. **Evaluación:** ROC-AUC, PR-AUC, precisión, recall y F1; ganancia por deciles; curvas de aprendizaje; SHAP sobre el Random Forest; importancia por permutación sobre el gradient boosting; intervalos de confianza por bootstrap; equidad por sexo y edad; calibración.
6. **Presentación:** informe técnico, resumen ejecutivo, PDF de 15 páginas y diapositivas.

---

## Consideraciones éticas

- **Datos de salud:** el dataset proviene de una encuesta de los CDC, sin nombres ni identificadores directos. Los resultados se presentan siempre en agregado y no deben usarse para identificar personas.
- **Uso del modelo:** las predicciones sirven para orientar la prevención y la conversación clínica. No reemplazan el diagnóstico médico ni deben usarse para decidir seguros, empleo o acceso a servicios.
- **Sesgo y equidad:** la encuesta es telefónica y autoreportada, por lo que puede no representar bien a todos los grupos. El desempeño por sexo y edad muestra diferencias de recall y precisión. Hay que revisarlas antes de cualquier uso real.
- **Falsos negativos:** con el umbral de 0,5, el modelo genera muchas alertas falsas. Esa carga debe considerarse frente al costo de pasar por alto un caso.
- **Licencia de los datos:** CC0 (dominio público). Su uso es libre con la atribución de la fuente.

---

## Limitaciones

- Datos autoreportados y de corte transversal: no permiten inferir causalidad.
- Sin búsqueda de hiperparámetros ni validación en otra población o en otro año de BRFSS.
- Las probabilidades no están calibradas.
- SHAP se calculó sobre 300 registros de prueba, por el costo computacional.

---

## Requisitos

Las dependencias están en [requirements.txt](requirements.txt): pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, shap, nbformat, nbconvert, nbclient, ipykernel, reportlab, pillow, python-docx y python-pptx.
