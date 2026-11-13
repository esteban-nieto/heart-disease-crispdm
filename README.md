# Enfermedad cardiaca con indicadores de salud (BRFSS 2015): CRISP-DM

Proyecto CRISP-DM sobre el dataset [Heart Disease Health Indicators (alexteboul)](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset), licencia CC0, basado en la encuesta BRFSS 2015 de los CDC.

## Estructura
- `data/`: CSV original, base SQLite `heart_disease.db` y datos limpios.
- `notebooks/`: scripts por fase (01 EDA, 02 preparación y BD, 03 modelamiento y evaluación).
- `reports/`: informe `resultados_crispdm.md`, tablas CSV y figuras.

## Ejecución (desde esta carpeta)
```bash
pip install -r requirements.txt
python notebooks/01_comprension_y_eda.py
python notebooks/02_preparacion_y_base_de_datos.py
python notebooks/03_modelamiento_y_evaluacion.py
```

## Consideraciones éticas

- **Datos de salud:** el dataset proviene de una encuesta de los CDC, sin nombres ni identificadores directos. Los resultados se presentan siempre en agregado y no deben usarse para identificar personas.
- **Uso del modelo:** las predicciones sirven para orientar la prevención y la conversación clínica. No reemplazan el diagnóstico médico ni deben usarse para decidir seguros, empleo o acceso a servicios.
- **Sesgo y equidad:** la encuesta es telefónica y autoreportada, por lo que puede no representar bien a todos los grupos. El desempeño por sexo y edad se revisó (sección 5.6), y muestra diferencias de recall y precisión. Hay que revisar esas brechas antes de cualquier uso real.
- **Falsos negativos:** un modelo con precisión baja y recall alto genera muchas alertas falsas. Esa carga debe considerarse frente a los costos de pasar por alto un caso.
- **Licencia:** el dataset se publica bajo CC0 (dominio público), por lo que su uso es libre con la atribución de la fuente.
