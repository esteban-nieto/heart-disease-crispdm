# %% [markdown]
# **Autor:** Esteban Alejandro Pedreros  
# **Fecha:** 13/11/2026
#
# %%
"""
CRISP-DM fases 1 y 2: comprensión del negocio y de los datos (EDA).

Ejecutar desde la raíz del proyecto:
    python notebooks/01_comprension_y_eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "heart_disease_health_indicators_BRFSS2015.csv"
FIGURAS = RAIZ / "reports" / "figures"
FIGURAS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATOS)
TARGET = "HeartDiseaseorAttack"
NUMERICAS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
BINARIAS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "Diabetes",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
            "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]
# Diabetes tiene códigos 0, 1 y 2; se trata como categórica en las pruebas.

# ---------- Fase 1: comprensión del negocio ----------
print("=== FASE 1: COMPRENSIÓN DEL NEGOCIO ===")
print("Objetivo: identificar factores de salud y hábitos asociados a infarto o "
      "enfermedad cardiaca (HeartDiseaseorAttack) y estimar la probabilidad de "
      "que una persona la presente, para apoyar la prevención.")
print("Fuente: encuesta BRFSS 2015 (CDC, EE. UU.), datos autoreportados.")

# ---------- Fase 2: comprensión de los datos ----------
print("\n=== FASE 2: COMPRENSIÓN DE LOS DATOS ===")
print("Filas x columnas:", df.shape)
print("Nulos totales:", int(df.isna().sum().sum()))
print("Duplicados:", int(df.duplicated().sum()))
print("\nBalance del target (con duplicados):")
print(df[TARGET].value_counts(normalize=True).round(4).to_string())

df_sin_dup = df.drop_duplicates()
print(f"\nFilas tras eliminar duplicados: {len(df_sin_dup)}")
print("Balance del target (sin duplicados):")
print(df_sin_dup[TARGET].value_counts(normalize=True).round(4).to_string())

resumen = df_sin_dup[NUMERICAS + BINARIAS].describe().T.round(2)
resumen.to_csv(RAIZ / "reports" / "estadisticas_descriptivas.csv")

# Distribuciones
fig, ejes = plt.subplots(2, 4, figsize=(16, 8))
for eje, col in zip(ejes.flat, NUMERICAS + ["Smoker"]):
    sns.histplot(df_sin_dup[col], bins=30, ax=eje, color="#2a7ab0")
    eje.set_title(col)
fig.tight_layout()
fig.savefig(FIGURAS / "01_distribuciones.png", dpi=110)
plt.close(fig)

# Asociación numéricas vs target
filas = []
for col in NUMERICAS:
    a = df_sin_dup.loc[df_sin_dup[TARGET] == 0, col]
    b = df_sin_dup.loc[df_sin_dup[TARGET] == 1, col]
    _, p = mannwhitneyu(a, b)
    filas.append({"variable": col, "media_sin_evento": round(a.mean(), 2),
                  "media_con_evento": round(b.mean(), 2),
                  "diferencia_std": round((b.mean() - a.mean()) / df_sin_dup[col].std(), 4),
                  "p_valor": p})
tabla_num = pd.DataFrame(filas)

# Asociación binarias vs target
filas = []
for col in BINARIAS:
    tabla = pd.crosstab(df_sin_dup[col], df_sin_dup[TARGET])
    _, p, _, _ = chi2_contingency(tabla)
    tasas = df_sin_dup.groupby(col)[TARGET].mean().round(4).to_dict()
    filas.append({"variable": col, "tasa_evento_por_nivel": tasas, "p_valor": p})
tabla_cat = pd.DataFrame(filas)

tabla_num.to_csv(RAIZ / "reports" / "asociacion_numericas_con_riesgo.csv", index=False)
tabla_cat.to_csv(RAIZ / "reports" / "asociacion_binarias_con_riesgo.csv", index=False)
print("\nAsociación numéricas vs target:")
print(tabla_num.to_string(index=False))
print("\nAsociación binarias vs target:")
print(tabla_cat.to_string(index=False))

# Correlación con el target
corr = df_sin_dup[NUMERICAS + BINARIAS + [TARGET]].corr()[TARGET].drop(TARGET)
corr.sort_values().to_csv(RAIZ / "reports" / "correlacion_con_target.csv", header=["r"])

fig, eje = plt.subplots(figsize=(9, 9))
sns.heatmap(df_sin_dup[NUMERICAS + BINARIAS + [TARGET]].corr(), cmap="RdBu_r",
            center=0, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=eje)
eje.set_title("Matriz de correlación (sin duplicados)")
fig.tight_layout()
fig.savefig(FIGURAS / "02_correlacion.png", dpi=110)
plt.close(fig)

# Tasa de evento por edad (código BRFSS 1-13) y por salud general
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(data=df_sin_dup.groupby("Age")[TARGET].mean().reset_index(),
            x="Age", y=TARGET, ax=ejes[0], color="#c0392b")
ejes[0].set_title("Tasa de infarto/enfermedad cardiaca por grupo de edad (código BRFSS)")
sns.barplot(data=df_sin_dup.groupby("GenHlth")[TARGET].mean().reset_index(),
            x="GenHlth", y=TARGET, ax=ejes[1], color="#8e44ad")
ejes[1].set_title("Tasa por salud general (1 = excelente, 5 = mala)")
fig.tight_layout()
fig.savefig(FIGURAS / "03_tasas_edad_salud.png", dpi=110)
plt.close(fig)

print("\nFiguras guardadas en reports/figures/")
