# %% [markdown]
# **Autor:** Esteban Alejandro Pedreros
# **Fecha:** 13/11/2026
#

# %% [markdown]
# # 6. Robustez y equidad (complemento de las fases 4 y 5)
#
# Reajusta la regresión logística y el Random Forest con la misma partición de la
# fase 3 y responde tres preguntas:
#
# 1. ¿Qué tan estables son las métricas? Intervalos de confianza por bootstrap
#    (1.000 remuestreos del conjunto de prueba).
# 2. ¿El modelo funciona igual para hombres y mujeres, y para mayores y menores de 65 años?
# 3. ¿Las probabilidades están calibradas? Brier score y tabla de calibración.
#
# Ejecutar desde la raíz del proyecto:
#     python notebooks/06_robustez_y_equidad.py

# %%
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
BD = RAIZ / "data" / "heart_disease.db"
REP = RAIZ / "reports"
FIG = REP / "figures"
SEMILLA = 42
B = 1000

# %% [markdown]
# ## 6.1. Misma partición que la fase 3

# %%
with sqlite3.connect(BD) as con:
    df = pd.read_sql("SELECT * FROM encuestados", con)

TARGET = "HeartDiseaseorAttack"
FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
            "Age", "Education", "Income"]
X = df[FEATURES]
y = df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
te = X_te.copy()
te["y"] = y_te.values

# %% [markdown]
# ## 6.2. Reajuste de los modelos principales

# %%
logit = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50,
                            class_weight="balanced_subsample", n_jobs=4,
                            random_state=SEMILLA)
logit.fit(X_tr, y_tr)
rf.fit(X_tr, y_tr)
p = {"Regresión logística": logit.predict_proba(X_te)[:, 1],
     "Random Forest": rf.predict_proba(X_te)[:, 1]}
print("Modelos reajustados.")

# %% [markdown]
# ## 6.3. Intervalos de confianza por bootstrap
#
# Se remuestrea el conjunto de prueba con reemplazo. El intervalo del 95 % va de
# los percentiles 2,5 y 97,5 de las métricas obtenidas.

# %%
rng = np.random.default_rng(SEMILLA)
idx = [rng.integers(0, len(y_te), len(y_te)) for _ in range(B)]
yt = y_te.values
filas = []
for nombre, pr in p.items():
    auc_b, pr_b = [], []
    for ii in idx:
        if yt[ii].min() == yt[ii].max():
            continue
        auc_b.append(roc_auc_score(yt[ii], pr[ii]))
        pr_b.append(average_precision_score(yt[ii], pr[ii]))
    for metrica, valor, boot in [("ROC-AUC", roc_auc_score(yt, pr), auc_b),
                                 ("PR-AUC", average_precision_score(yt, pr), pr_b)]:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        filas.append({"modelo": nombre, "metrica": metrica,
                      "valor": round(valor, 4),
                      "ic95_inferior": round(lo, 4),
                      "ic95_superior": round(hi, 4)})
tabla_ic = pd.DataFrame(filas)
print(tabla_ic.to_string(index=False))
tabla_ic.to_csv(REP / "intervalos_confianza_bootstrap.csv", index=False)

# %% [markdown]
# ## 6.4. Equidad: métricas por subgrupo
#
# Se calculan ROC-AUC, recall y precisión (umbral 0,5) para cada grupo. Una brecha
# grande entre grupos indica que el modelo no funciona igual para todos.

# %%
def metricas_grupo(mascara, pr):
    yy = te.loc[mascara, "y"].values
    pp = pr[mascara.values]
    pred = (pp >= 0.5).astype(int)
    tp = int(((pred == 1) & (yy == 1)).sum())
    fp = int(((pred == 1) & (yy == 0)).sum())
    fn = int(((pred == 0) & (yy == 1)).sum())
    return {
        "n": len(yy),
        "prevalencia": round(yy.mean(), 4),
        "roc_auc": round(roc_auc_score(yy, pp), 4) if len(np.unique(yy)) > 1 else np.nan,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else np.nan,
    }

grupos = {
    "Sexo: hombre (1)": te["Sex"] == 1,
    "Sexo: mujer (0)": te["Sex"] == 0,
    "Edad: 65 años o más (código 10 a 13)": te["Age"] >= 10,
    "Edad: menores de 65 (código 1 a 9)": te["Age"] < 10,
}
filas = []
for nombre, pr in p.items():
    for gnombre, m in grupos.items():
        filas.append({"modelo": nombre, "grupo": gnombre, **metricas_grupo(m, pr)})
tabla_eq = pd.DataFrame(filas)
print(tabla_eq.to_string(index=False))
tabla_eq.to_csv(REP / "metricas_por_subgrupo.csv", index=False)

# %% [markdown]
# ## 6.5. Calibración
#
# Brier score (menor es mejor) y tabla de calibración por deciles de probabilidad
# predicha. Si la tasa observada se parece a la predicha, las probabilidades son
# confiables para usarlas como riesgo.

# %%
filas = []
for nombre, pr in p.items():
    filas.append({"modelo": nombre,
                  "brier": round(brier_score_loss(yt, pr), 4),
                  "prevalencia_media_predicha": round(pr.mean(), 4),
                  "prevalencia_observada": round(yt.mean(), 4)})
tabla_brier = pd.DataFrame(filas)
print(tabla_brier.to_string(index=False))
tabla_brier.to_csv(REP / "calibracion_brier.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for nombre, pr in p.items():
    deciles = pd.qcut(pr, 10, labels=False, duplicates="drop")
    d = pd.DataFrame({"pred": pr, "obs": yt, "dec": deciles}).groupby("dec").mean()
    ax.plot(d["pred"], d["obs"], marker="o", label=nombre)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibración perfecta")
ax.set_xlabel("Probabilidad predicha (media por decil)")
ax.set_ylabel("Tasa observada")
ax.set_title("Curva de calibración en prueba")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "09_calibracion.png", dpi=130)
plt.close(fig)
print("Figura guardada: 09_calibracion.png")
