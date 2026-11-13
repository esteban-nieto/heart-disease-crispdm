# Enfermedad cardiaca: factores de riesgo y modelos predictivos (CRISP-DM)

Proyecto de ciencia de datos que estudia qué factores de salud y hábitos se asocian con infarto o enfermedad cardiaca, y qué tan bien se puede estimar ese riesgo con modelos predictivos. Sigue la metodología CRISP-DM en sus seis fases.

**Autor:** Esteban Alejandro Pedreros
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
│   ├── 01_comprension_y_eda.py / .ipynb                Fases 1 y 2: negocio, datos, EDA
│   ├── 02_preparacion_y_base_de_datos.py / .ipynb      Fase 3: validación, limpieza, SQLite
│   ├── 03_modelamiento_y_evaluacion.py / .ipynb        Fases 4 y 5: modelos, SHAP, deciles
│   ├── 04_presentacion_pdf.py                          Genera el PDF de presentación
│   ├── 05_formato_excel_y_word.py                      Reformatea CSV para Excel y genera Word
│   ├── 06_robustez_y_equidad.py / .ipynb               Complemento: bootstrap, equidad, calibración
│   ├── 07_diapositivas_pptx.py                         Genera las diapositivas
│   └── _convertir_a_ipynb.py                           Convierte los scripts en notebooks
├── reports/
│   ├── informe_tecnico.md / .docx                      Informe técnico paso a paso
│   ├── resultados_crispdm.md / .docx                   Resumen ejecutivo
│   ├── presentacion_resultados.pdf                     Presentación (15 páginas)
│   ├── presentacion_resultados.pptx                    Diapositivas
│   ├── *.csv                                           Tablas de métricas y análisis
│   ├── figures/                                        Figuras 01 a 09
│   ├── screenshots/                                    Pantallazos de terminal, notebook y Kaggle
│   └── logs/                                           Salida completa de cada ejecución
├── .gitignore
├── requirements.txt
└── README.md
```

La base de datos `data/heart_disease.db` no está en el repositorio: la crea el script 02 a partir del CSV original.

---

## Cómo reproducir el análisis

Se recomienda Python 3.11 o superior. Desde la raíz de esta carpeta:

```bash
pip install -r requirements.txt
python notebooks/01_comprension_y_eda.py
python notebooks/02_preparacion_y_base_de_datos.py
python notebooks/03_modelamiento_y_evaluacion.py
python notebooks/06_robustez_y_equidad.py
python notebooks/04_presentacion_pdf.py
python notebooks/05_formato_excel_y_word.py
python notebooks/07_diapositivas_pptx.py
```

Notas:
- El script 03 tarda unos 30 minutos, sobre todo por SHAP. Se usa una muestra de 300 registros para SHAP.
- El script 04 necesita que existan las tablas de las fases 5 y 6. El orden anterior ya lo respeta.
- El script 05 reescribe los CSV en formato `;` y decimal `,` para Excel en configuración española, y convierte los Markdown a Word. Debe ejecutarse después de los demás, y no mientras el Word esté abierto.
- Para regenerar los notebooks: `python notebooks/_convertir_a_ipynb.py 01_comprension_y_eda 02_preparacion_y_base_de_datos 06_robustez_y_equidad`. El notebook 03 se ejecuta aparte, porque tarda y ya tiene sus salidas guardadas.

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
