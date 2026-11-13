"""
CRISP-DM fase 6: presentación ejecutiva en PDF.

Genera reports/presentacion_resultados.pdf a partir de las tablas y figuras
producidas por las fases 1 a 5. No recalcula nada: solo presenta resultados.

Ejecutar desde la raíz del proyecto:
    python notebooks/04_presentacion_pdf.py
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
REP = RAIZ / "reports"
FIG = REP / "figures"
SAL = REP / "presentacion_resultados.pdf"

estilos = getSampleStyleSheet()
titulo = ParagraphStyle("t", parent=estilos["Title"], fontSize=26, leading=30)
h1 = ParagraphStyle("h1", parent=estilos["Heading1"], fontSize=20, spaceAfter=8)
texto = ParagraphStyle("x", parent=estilos["BodyText"], fontSize=12, leading=16)
pequeno = ParagraphStyle("p", parent=estilos["BodyText"], fontSize=9, leading=11,
                         textColor=colors.HexColor("#555555"))


def tabla(df):
    datos = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(datos, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a7ab0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
    ]))
    return t


def imagen(ruta, ancho_cm=20, alto_max_cm=15.5):
    """Escala la imagen para que quepa en la página horizontal (ancho y alto)."""
    from PIL import Image as PILImage
    with PILImage.open(ruta) as im:
        w, h = im.size
    ancho = ancho_cm * cm
    alto = ancho * h / w
    if alto > alto_max_cm * cm:
        alto = alto_max_cm * cm
        ancho = alto * w / h
    return Image(str(ruta), width=ancho, height=alto)


def pagina_figura(titulo_txt, ruta, nota):
    return [Paragraph(titulo_txt, h1), imagen(ruta, 19 if "pantallazo" not in nota else 19),
            Spacer(1, 6), Paragraph(nota, pequeno), PageBreak()]


def leer(nombre):
    """Lee el CSV en formato coma/punto o ;/coma (05 lo reescribe en ;/coma)."""
    ruta = REP / nombre
    if ";" in ruta.read_text(encoding="utf-8-sig").splitlines()[0]:
        return pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(ruta, encoding="utf-8-sig")


cv = leer("metricas_validacion_cruzada.csv")
test = leer("metricas_test.csv")
decil = leer("ganancia_por_decil_random_forest.csv")

historia = []

# Portada
historia += [Spacer(1, 4 * cm),
             Paragraph("Enfermedad cardiaca: factores de riesgo y modelos predictivos", titulo),
             Spacer(1, 12),
             Paragraph("Metodología CRISP-DM · Encuesta BRFSS 2015 (CDC) · "
                       "229.781 registros sin duplicados", texto),
             Spacer(1, 30),
             Paragraph("Fuente: Heart Disease Health Indicators (alexteboul), Kaggle, "
                       "licencia CC0.", pequeno),
             PageBreak()]

# 1. Negocio
historia += [Paragraph("1. Comprensión del negocio", h1),
             Paragraph("Pregunta: ¿qué factores de salud y hábitos se asocian con infarto o "
                       "enfermedad cardiaca, y qué tan bien se puede estimar ese riesgo para "
                       "apoyar la prevención?", texto),
             Spacer(1, 8),
             Paragraph("Límite del estudio: datos autoreportados y de corte transversal. "
                       "Muestran asociaciones, no causas.", texto),
             PageBreak()]

# 2. Datos
historia += [Paragraph("2. Comprensión y preparación de los datos", h1),
             Paragraph("253.680 registros originales; 23.899 duplicados exactos eliminados; "
                       "229.781 registros finales. Sin nulos. Evento en 10,3 % (desbalance).",
                       texto),
             Spacer(1, 8),
             Paragraph("Codificación: binarias 0/1; ordinales (edad, salud general, "
                       "educación, ingreso, diabetes) conservadas como enteros para respetar su "
                       "orden. Normalización: StandardScaler solo para regresión logística, "
                       "ajustado con entrenamiento.", texto),
             PageBreak()]
historia += pagina_figura("2.1 Distribuciones", FIG / "01_distribuciones.png",
                          "Distribución de variables numéricas tras la limpieza.")
historia += pagina_figura("2.2 Tasa de evento por edad y salud general",
                          FIG / "03_tasas_edad_salud.png",
                          "La tasa sube de 0,5 % (grupo 1) a 24,6 % (grupo 13).")

# 3. Modelos
historia += [Paragraph("3. Modelamiento y evaluación", h1),
             Paragraph("Validación cruzada de 5 folds (entrenamiento):", texto),
             Spacer(1, 4), tabla(cv), Spacer(1, 12),
             Paragraph("Prueba (20 %, 45.957 registros):", texto),
             Spacer(1, 4), tabla(test), PageBreak()]
historia += pagina_figura("3.1 Curvas ROC en prueba", FIG / "04_curvas_roc.png",
                          "Todos los modelos superan la línea base (AUC 0,50).")
historia += pagina_figura("3.2 Ganancia por deciles (Random Forest)",
                          FIG / "06_ganancia_por_decil.png",
                          "El decil de mayor riesgo predicho concentra la mayor tasa de evento.")
historia += pagina_figura("3.3 Curvas de aprendizaje", FIG / "07_curvas_aprendizaje.png",
                          "Brecha entre entrenamiento y validación: diagnóstico de sobreajuste.")
if (FIG / "08_shap_resumen.png").exists():
    historia += pagina_figura("3.4 Explicabilidad (SHAP, Random Forest)",
                              FIG / "08_shap_resumen.png",
                              "Valores SHAP de una muestra de 300 registros de prueba.")
historia += pagina_figura("3.5 Matriz de confusión (umbral 0,5)",
                          FIG / "05_matriz_confusion.png",
                          "Gradient boosting con umbral 0,5: detecta ~80 % de los casos (3.801 de 4.743), con precisión ~25 %.")

# 3.6 a 3.8 Robustez, equidad y calibración (fase 6)
ic = leer("intervalos_confianza_bootstrap.csv")
ic["IC 95 %"] = ic["ic95_inferior"].map(lambda v: f"{v:.4f}") + " a " + ic["ic95_superior"].map(lambda v: f"{v:.4f}")
ic = ic[["modelo", "metrica", "valor", "IC 95 %"]]
ic.columns = ["Modelo", "Métrica", "Valor", "IC 95 % (bootstrap)"]
eq = leer("metricas_por_subgrupo.csv")
eq = eq[["modelo", "grupo", "n", "roc_auc", "recall", "precision"]]
eq.columns = ["Modelo", "Grupo", "n", "ROC-AUC", "Recall", "Precisión"]
cal = leer("calibracion_brier.csv")
cal.columns = ["Modelo", "Brier", "Prob. media predicha", "Prevalencia observada"]
historia += [Paragraph("3.6 Robustez: intervalos de confianza", h1),
             Paragraph("Bootstrap sobre el conjunto de prueba, 1.000 remuestreos. Los intervalos "
                       "se traslapan: la diferencia entre modelos no es distinguible.", texto),
             Spacer(1, 6), tabla(ic), PageBreak()]
historia += [Paragraph("3.7 Equidad: métricas por sexo y edad", h1),
             Paragraph("Umbral 0,5. El recall es mayor en mujeres y en mayores de 65, pero la "
                       "precisión es menor. El ROC-AUC en mayores de 65 es cercano a 0,76.", texto),
             Spacer(1, 6), tabla(eq), PageBreak()]
historia += [Paragraph("3.8 Calibración de las probabilidades", h1),
             Paragraph("Las probabilidades no están calibradas: su promedio es cercano a 0,36 "
                       "frente a una prevalencia de 0,10 por el efecto de class_weight='balanced'. "
                       "Sirven para ordenar el riesgo, no como probabilidad absoluta.", texto),
             Spacer(1, 6), tabla(cal), Spacer(1, 10),
             imagen(FIG / "09_calibracion.png", 14, 11), PageBreak()]

# 4. Conclusiones
historia += [Paragraph("4. Conclusiones y recomendaciones", h1),
             Paragraph("• Edad y salud general son los factores más asociados al evento.", texto),
             Paragraph("• Regresión logística y Random Forest rinden casi igual (AUC ≈ 0,84). "
                       "La regresión es más simple de explicar.", texto),
             Paragraph("• El umbral debe elegirse según el costo de falsos positivos frente a "
                       "falsos negativos en la prevención.", texto),
             Paragraph("• Equidad y calibración: el modelo no rinde igual por grupo y sus "
                       "probabilidades no son absolutas. Revisar antes de usarlo.", texto),
             Paragraph("• Limitaciones: datos autoreportados, sin búsqueda de hiperparámetros "
                       "y sin validación externa.", texto)]


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[1] - 1.5 * cm, 0.8 * cm, f"Página {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(SAL), pagesize=landscape(A4),
                        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                        title="Presentación de resultados: enfermedad cardiaca (CRISP-DM)")
doc.build(historia, onFirstPage=pie, onLaterPages=pie)
print("PDF generado:", SAL)
