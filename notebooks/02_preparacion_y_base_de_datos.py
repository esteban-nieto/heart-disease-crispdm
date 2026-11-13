# %% [markdown]
# **Autor:** Esteban Alejandro Pedreros  
# **Fecha:** 13/11/2026
#
# %%
"""
CRISP-DM fase 3: preparación de los datos y carga en base de datos SQLite.

Crea data/heart_disease.db con dos tablas:
  - raw_encuesta:  datos originales (253.680 filas).
  - encuestados:   datos sin duplicados, validados y con variables derivadas.

Ejecutar desde la raíz del proyecto:
    python notebooks/02_preparacion_y_base_de_datos.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data" / "heart_disease_health_indicators_BRFSS2015.csv"
BD = RAIZ / "data" / "heart_disease.db"

df_raw = pd.read_csv(CSV)

# ---------- Validación de códigos y rangos (según el libro de códigos BRFSS) ----------
RANGOS = {
    "BMI": (12, 98), "MentHlth": (0, 30), "PhysHlth": (0, 30),
    "GenHlth": (1, 5), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8),
    "Diabetes": (0, 2),
}
BINARIAS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "PhysActivity",
            "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "DiffWalk", "Sex", "HeartDiseaseorAttack"]

errores = []
for col, (lo, hi) in RANGOS.items():
    fuera = int(((df_raw[col] < lo) | (df_raw[col] > hi)).sum())
    if fuera:
        errores.append(f"{col}: {fuera} valores fuera de [{lo}, {hi}]")
for col in BINARIAS:
    if not set(df_raw[col].dropna().unique()) <= {0, 1}:
        errores.append(f"{col}: valores distintos de 0/1")
print("Validación:", "OK" if not errores else errores)

# ---------- Limpieza ----------
df = df_raw.drop_duplicates().reset_index(drop=True)
print(f"Filas originales: {len(df_raw)}  |  sin duplicados: {len(df)}")

# ---------- Variables derivadas ----------
# Código de edad BRFSS: 1 = 18-24 años, ..., 13 = 80 años o más (grupos de 5 años).
EDAD_APROX = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57,
              9: 62, 10: 67, 11: 72, 12: 77, 13: 82}
df["edad_aprox"] = df["Age"].map(EDAD_APROX)
df["categoria_imc"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 1000],
                             labels=["bajo_peso", "normal", "sobrepeso", "obesidad"],
                             right=False).astype(str)
df["num_condiciones"] = df[["HighBP", "HighChol", "Diabetes", "Smoker", "Stroke"]].gt(0).sum(axis=1)
df["inactivo"] = (1 - df["PhysActivity"]).astype(int)
df["sexo"] = df["Sex"].map({0: "Mujer", 1: "Hombre"})

# ---------- Carga en SQLite ----------
if BD.exists():
    BD.unlink()
with sqlite3.connect(BD) as con:
    df_raw.to_sql("raw_encuesta", con, index=False)
    df.to_sql("encuestados", con, index=False)
    con.execute("CREATE INDEX idx_enc_target ON encuestados(HeartDiseaseorAttack)")
    con.execute("CREATE INDEX idx_enc_edad ON encuestados(Age)")
    con.commit()

with sqlite3.connect(BD) as con:
    n = con.execute("SELECT COUNT(*) FROM encuestados").fetchone()[0]
    tasas = pd.read_sql(
        "SELECT Age AS grupo_edad_brfss, ROUND(AVG(HeartDiseaseorAttack),4) AS tasa_evento, "
        "COUNT(*) AS n FROM encuestados GROUP BY Age ORDER BY Age", con)
print(f"\nBD creada en {BD} con {n} filas en 'encuestados'.")
print(tasas.to_string(index=False))

