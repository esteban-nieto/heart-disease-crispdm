"""
Diapositivas de los resultados en formato PowerPoint.

Lee las tablas y figuras de reports/ (no recalcula nada) y genera
reports/presentacion_resultados.pptx.

Ejecutar desde la raíz del proyecto:
    python notebooks/07_diapositivas_pptx.py
"""
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Cm, Pt

RAIZ = Path(__file__).resolve().parents[1]
REP = RAIZ / "reports"
FIG = REP / "figures"
SAL = REP / "presentacion_resultados.pptx"

AZUL = RGBColor(0x2A, 0x7A, 0xB0)
GRIS = RGBColor(0x55, 0x55, 0x55)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)


def leer(nombre):
    ruta = REP / nombre
    if ";" in ruta.read_text(encoding="utf-8-sig").splitlines()[0]:
        return pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(ruta, encoding="utf-8-sig")


prs = Presentation()
prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)
VACIA = prs.slide_layouts[6]


def texto(diap, x, y, w, h, lineas, tam=18, color=None, negrita=False):
    caja = diap.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    marco = caja.text_frame
    marco.word_wrap = True
    for i, linea in enumerate(lineas):
        parrafo = marco.paragraphs[0] if i == 0 else marco.add_paragraph()
        parrafo.text = linea
        parrafo.font.size = Pt(tam)
        parrafo.font.bold = negrita
        parrafo.font.color.rgb = color or RGBColor(0x20, 0x20, 0x20)
        parrafo.space_after = Pt(8)
    return caja


def diapositiva(titulo):
    diap = prs.slides.add_slide(VACIA)
    texto(diap, 1.5, 0.8, 30.5, 2.2, [titulo], tam=30, color=AZUL, negrita=True)
    return diap


def tabla(diap, df, x, y, w, tam=14):
    filas, cols = len(df) + 1, len(df.columns)
    forma = diap.shapes.add_table(filas, cols, Cm(x), Cm(y), Cm(w), Cm(0.9 * filas))
    t = forma.table
    for j, nombre in enumerate(df.columns):
        celda = t.cell(0, j)
        celda.text = str(nombre)
        celda.fill.solid()
        celda.fill.fore_color.rgb = AZUL
        for p in celda.text_frame.paragraphs:
            p.font.size, p.font.bold, p.font.color.rgb = Pt(tam), True, BLANCO
    for i, fila in enumerate(df.itertuples(index=False), start=1):
        for j, valor in enumerate(fila):
            celda = t.cell(i, j)
            celda.text = str(valor)
            for p in celda.text_frame.paragraphs:
                p.font.size = Pt(tam)
    return forma


def figura(diap, ruta, x, y, alto):
    diap.shapes.add_picture(str(ruta), Cm(x), Cm(y), height=Cm(alto))


# 1. Portada
diap = prs.slides.add_slide(VACIA)
texto(diap, 2, 5.5, 30, 3, ["Enfermedad cardiaca: factores de riesgo y modelos predictivos"],
      tam=36, color=AZUL, negrita=True)
texto(diap, 2, 10, 30, 2, ["Metodología CRISP-DM  ·  Encuesta BRFSS 2015 (CDC)  ·  229.781 registros"],
      tam=20, color=GRIS)
texto(diap, 2, 15, 30, 2, ["Fuente: Heart Disease Health Indicators (alexteboul), Kaggle, licencia CC0"],
      tam=14, color=GRIS)

# 2. Pregunta de negocio
diap = diapositiva("1. Comprensión del negocio")
texto(diap, 2, 4, 29, 12, [
    "Pregunta: ¿qué factores de salud y hábitos se asocian con infarto o enfermedad cardiaca, "
    "y qué tan bien se puede estimar ese riesgo para apoyar la prevención?",
    "Límite del estudio: datos autoreportados y de corte transversal. Muestran asociaciones, no causas.",
], tam=22)

# 3. Datos
diap = diapositiva("2. Datos y preparación")
texto(diap, 2, 4, 29, 6, [
    "253.680 registros originales; 23.899 duplicados exactos eliminados; 229.781 finales.",
    "Sin nulos. Evento (infarto o enfermedad cardiaca) en 10,3 %: desbalance.",
    "Binarias 0/1. Ordinales (edad, salud general, educación, ingreso, diabetes) como enteros.",
    "StandardScaler solo para regresión logística, ajustado con el entrenamiento.",
], tam=20)
figura(diap, FIG / "03_tasas_edad_salud.png", 2, 11, 7)

# 4. Factores
diap = diapositiva("3. Factores asociados")
texto(diap, 2, 4, 12, 12, [
    "Edad y salud general son los factores más fuertes.",
    "La tasa de evento sube de 0,5 % (edad grupo 1) a 24,6 % (grupo 13).",
    "Por SHAP, los más influyentes son edad, presión alta, salud general y colesterol alto.",
], tam=20)
figura(diap, FIG / "08_shap_resumen.png", 15, 3, 15)

# 5. Modelos
cv = leer("metricas_validacion_cruzada.csv")
test = leer("metricas_test.csv")
diap = diapositiva("4. Modelos y resultados en prueba")
tabla(diap, test.rename(columns={"modelo": "Modelo", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
      "precision_umbral_0.5": "Precisión", "recall_umbral_0.5": "Recall",
      "f1_umbral_0.5": "F1"}), 2, 4, 29.5, tam=16)
texto(diap, 2, 12, 29, 4, [
    "Regresión logística y Random Forest rinden casi igual (ROC-AUC cercano a 0,84).",
    "Gradient boosting se agrega como comparación. La línea base tiene ROC-AUC de 0,50.",
], tam=18)

# 6. Curvas ROC
diap = diapositiva("5. Curvas ROC en prueba")
figura(diap, FIG / "04_curvas_roc.png", 4, 3.5, 13)
texto(diap, 2, 17, 29, 1.5, ["Todos los modelos superan la línea base (AUC 0,50)."], tam=16, color=GRIS)

# 7. Deciles
diap = diapositiva("6. Ganancia por deciles (Random Forest)")
figura(diap, FIG / "06_ganancia_por_decil.png", 4, 3.5, 13)
texto(diap, 2, 17, 29, 1.5, [
    "El decil de mayor riesgo predicho concentra una tasa de evento de 40 %, frente a 10 % en promedio."
], tam=16, color=GRIS)

# 8. Robustez
ic = leer("intervalos_confianza_bootstrap.csv")
ic["IC 95 %"] = (ic["ic95_inferior"].map(lambda v: f"{v:.4f}") + " a "
                 + ic["ic95_superior"].map(lambda v: f"{v:.4f}"))
ic = ic[["modelo", "metrica", "valor", "IC 95 %"]]
ic.columns = ["Modelo", "Métrica", "Valor", "IC 95 %"]
diap = diapositiva("7. Robustez: intervalos de confianza")
tabla(diap, ic, 2, 4, 29.5, tam=18)
texto(diap, 2, 12, 29, 4, [
    "Bootstrap sobre prueba, 1.000 remuestreos. Los intervalos se traslapan.",
    "La diferencia entre regresión logística y Random Forest no es distinguible.",
], tam=18)

# 9. Equidad
eq = leer("metricas_por_subgrupo.csv")
eq = eq[["modelo", "grupo", "n", "roc_auc", "recall", "precision"]]
eq.columns = ["Modelo", "Grupo", "n", "ROC-AUC", "Recall", "Precisión"]
diap = diapositiva("8. Equidad: sexo y edad")
tabla(diap, eq, 2, 4, 29.5, tam=14)
texto(diap, 2, 15.5, 29, 3, [
    "El recall es mayor en mujeres y en mayores de 65, pero la precisión es menor.",
    "El ROC-AUC en mayores de 65 es cercano a 0,76. Revisar antes de cualquier uso real.",
], tam=16)

# 10. Calibración
diap = diapositiva("9. Calibración de las probabilidades")
figura(diap, FIG / "09_calibracion.png", 2, 3.5, 11)
texto(diap, 17, 4, 15, 12, [
    "Brier: 0,171 (regresión logística) y 0,165 (Random Forest).",
    "El promedio predicho es cercano a 0,36; la prevalencia real es 0,10.",
    "Las probabilidades sirven para ordenar riesgo, no como probabilidad absoluta.",
], tam=18)

# 11. Conclusiones
diap = diapositiva("10. Conclusiones y limitaciones")
texto(diap, 2, 4, 29, 13, [
    "Edad, salud general, sexo, colesterol y presión alta son los factores más asociados al evento.",
    "El modelo ordena el riesgo con ROC-AUC cercano a 0,84, pero no rinde igual por grupo.",
    "El umbral debe elegirse según el costo de falsos positivos frente a falsos negativos.",
    "Limitaciones: datos autoreportados, sin búsqueda de hiperparámetros, sin validación externa "
    "y probabilidades sin calibrar.",
], tam=20)

prs.save(SAL)
print("PPTX generado:", SAL)
