# %% [markdown]
# **Autor:** Esteban Alejandro Pedreros  
# **Fecha:** 13/11/2026
#

# %% [markdown]
# # 4. Modelamiento y 5. Evaluación
#
# Lee la tabla `encuestados` desde `data/heart_disease.db`, compara modelos con
# validación cruzada y evalúa en un conjunto de prueba (20 %).
#
# Modelos: línea base, **Regresión logística** y **Random Forest** (el mismo
# par de familias que usa el repositorio de referencia), más un
# **gradient boosting** como comparación adicional.
#
# Explicabilidad: **SHAP** sobre el Random Forest. Diagnóstico: **curvas de
# aprendizaje** y **curva de ganancia por deciles**.
#
# Ejecutar desde la raíz del proyecto:
#     python notebooks/03_modelamiento_y_evaluacion.py

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
import shap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
BD = RAIZ / "data" / "heart_disease.db"
FIGURAS = RAIZ / "reports" / "figures"
FIGURAS.mkdir(parents=True, exist_ok=True)
SEMILLA = 42

# %% [markdown]
# ## 4.1. Datos y partición

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
print(f"Entrenamiento: {len(X_tr)}  |  prueba: {len(X_te)}  |  "
      f"prevalencia en prueba: {y_te.mean():.4f}")

# %% [markdown]
# ## 4.2. Construcción de los modelos
#
# - **Regresión logística:** requiere escalado (`StandardScaler`) porque penaliza
#   por igual todas las variables.
# - **Random Forest y gradient boosting:** basados en árboles, no requieren escalado.
# - `class_weight="balanced"` compensa el desbalance (10 % de positivos).

# %%
modelos = {
    "Línea base (mayoritaria)": make_pipeline(
        StandardScaler(), DummyClassifier(strategy="most_frequent")),
    "Regresión logística": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=50,
        class_weight="balanced_subsample", n_jobs=4, random_state=SEMILLA),
    "Gradient boosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=SEMILLA),
}

# %% [markdown]
# ## 4.3. Validación cruzada (solo entrenamiento, 5 folds)

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
filas = []
for nombre, modelo in modelos.items():
    r = cross_validate(modelo, X_tr, y_tr, cv=cv,
                       scoring=["roc_auc", "average_precision", "f1"], n_jobs=1)
    filas.append({
        "modelo": nombre,
        "cv_roc_auc": round(r["test_roc_auc"].mean(), 4),
        "cv_roc_auc_std": round(r["test_roc_auc"].std(), 4),
        "cv_pr_auc": round(r["test_average_precision"].mean(), 4),
        "cv_f1": round(r["test_f1"].mean(), 4),
    })
tabla_cv = pd.DataFrame(filas)
print("=== VALIDACIÓN CRUZADA (5 folds) ===")
print(tabla_cv.to_string(index=False))
tabla_cv.to_csv(RAIZ / "reports" / "metricas_validacion_cruzada.csv", index=False)

# %% [markdown]
# ## 5. Evaluación en el conjunto de prueba

# %%
filas = []
probs = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_tr, y_tr)
    prob = modelo.predict_proba(X_te)[:, 1]
    probs[nombre] = prob
    pred = (prob >= 0.5).astype(int)
    filas.append({
        "modelo": nombre,
        "roc_auc": round(roc_auc_score(y_te, prob), 4),
        "pr_auc": round(average_precision_score(y_te, prob), 4),
        "precision_umbral_0.5": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall_umbral_0.5": round(recall_score(y_te, pred, zero_division=0), 4),
        "f1_umbral_0.5": round(f1_score(y_te, pred, zero_division=0), 4),
    })
tabla_test = pd.DataFrame(filas)
print("\n=== EVALUACIÓN EN PRUEBA (20 %) ===")
print(tabla_test.to_string(index=False))
tabla_test.to_csv(RAIZ / "reports" / "metricas_test.csv", index=False)

# %%
# Curvas ROC
fig, eje = plt.subplots(figsize=(6, 5))
for nombre, prob in probs.items():
    RocCurveDisplay.from_predictions(y_te, prob, name=nombre, ax=eje)
eje.plot([0, 1], [0, 1], "k--", lw=0.8)
eje.set_title("Curvas ROC en prueba")
fig.tight_layout()
fig.savefig(FIGURAS / "04_curvas_roc.png", dpi=110)
plt.close(fig)

# Matriz de confusión del mejor modelo por ROC-AUC
mejor = tabla_test[tabla_test["modelo"] != "Línea base (mayoritaria)"] \
    .sort_values("roc_auc", ascending=False).iloc[0]["modelo"]
pred_mejor = (probs[mejor] >= 0.5).astype(int)
fig, eje = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_te, pred_mejor, ax=eje, cmap="Blues",
                                        display_labels=["Sin evento", "Con evento"])
eje.set_title(f"Matriz de confusión (umbral 0,5)\n{mejor}")
fig.tight_layout()
fig.savefig(FIGURAS / "05_matriz_confusion.png", dpi=110)
plt.close(fig)

# %% [markdown]
# ### 5.1. Ganancia por deciles
# Se ordenan los encuestados de prueba por probabilidad predicha y se mide la tasa
# real de evento en cada decil. Permite priorizar a quién contactar primero.

# %%
decil_df = pd.DataFrame({"prob": probs["Random Forest"], "y": y_te.to_numpy()})
decil_df["decil"] = pd.qcut(decil_df["prob"].rank(method="first", ascending=False),
                            10, labels=range(1, 11))
ganancia = decil_df.groupby("decil", observed=True)["y"].agg(["mean", "size"]) \
    .rename(columns={"mean": "tasa_evento", "size": "n"})
ganancia["lift"] = ganancia["tasa_evento"] / y_te.mean()
ganancia.round(4).to_csv(RAIZ / "reports" / "ganancia_por_decil_random_forest.csv")
print("\nTasa de evento por decil (Random Forest, 1 = mayor riesgo predicho):")
print(ganancia.round(4).to_string())

fig, eje = plt.subplots(figsize=(7, 4))
eje.bar(ganancia.index.astype(int), ganancia["tasa_evento"], color="#c0392b")
eje.axhline(y_te.mean(), color="k", ls="--", lw=0.8, label="tasa media")
eje.set_xlabel("Decil de riesgo predicho (1 = mayor)")
eje.set_ylabel("Tasa real de evento")
eje.set_title("Ganancia por deciles (Random Forest, prueba)")
eje.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "06_ganancia_por_decil.png", dpi=110)
plt.close(fig)

# %% [markdown]
# ## 5.2. Curvas de aprendizaje
# Diagnóstico de sobreajuste: si la brecha entre entrenamiento y validación es
# grande, el modelo memoriza; si ambas curvas están cerca, generaliza bien.

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
for eje, nombre in zip(ejes, ["Regresión logística", "Random Forest"]):
    tam, tr_sc, va_sc = learning_curve(
        modelos[nombre], X_tr, y_tr, cv=3, scoring="average_precision",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1, random_state=SEMILLA)
    eje.plot(tam, tr_sc.mean(axis=1), "o-", label="entrenamiento")
    eje.plot(tam, va_sc.mean(axis=1), "o-", label="validación")
    eje.set_title(f"Curva de aprendizaje: {nombre}")
    eje.set_xlabel("Tamaño de entrenamiento")
    eje.set_ylabel("PR-AUC")
    eje.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "07_curvas_aprendizaje.png", dpi=110)
plt.close(fig)

# %% [markdown]
# ## 5.3. Explicabilidad con SHAP (Random Forest)
# Se usa una muestra de 300 registros de prueba: con 300 árboles de profundidad 12,
# TreeExplainer tarda demasiado con muestras grandes. check_additivity se desactiva
# para acelerar; la suma de SHAP se verificó en una corrida exploratoria.

# %%
rf = modelos["Random Forest"]
muestra = X_te.sample(300, random_state=SEMILLA)
explicador = shap.TreeExplainer(rf)
valores = explicador.shap_values(muestra, check_additivity=False)
# En algunas versiones devuelve una lista por clase; se toma la clase positiva.
if isinstance(valores, list):
    valores = valores[1]
elif valores.ndim == 3:
    valores = valores[:, :, 1]

fig = plt.figure()
shap.summary_plot(valores, muestra, show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIGURAS / "08_shap_resumen.png", dpi=110, bbox_inches="tight")
plt.close("all")

importancia_shap = pd.Series(np.abs(valores).mean(axis=0), index=FEATURES) \
    .sort_values(ascending=False)
importancia_shap.to_csv(RAIZ / "reports" / "importancia_shap_random_forest.csv",
                        header=["abs_shap_medio"])
print("\nImportancia SHAP media absoluta (Random Forest), top 10:")
print(importancia_shap.head(10).round(4).to_string())

# %% [markdown]
# ## 5.4. Importancia por permutación (gradient boosting, comparación)

# %%
gb = modelos["Gradient boosting"]
perm = permutation_importance(gb, X_te, y_te, scoring="average_precision",
                              n_repeats=3, random_state=SEMILLA, n_jobs=1)
imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
imp.to_csv(RAIZ / "reports" / "importancia_permutacion_gb.csv", header=["reduccion_pr_auc"])
print("\nImportancia por permutación (gradient boosting), top 10:")
print(imp.head(10).round(4).to_string())

# %% [markdown]
# ## 5.5. Coeficientes de la regresión logística

# %%
logit = modelos["Regresión logística"]
coef = pd.Series(logit.named_steps["logisticregression"].coef_[0], index=FEATURES)
coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
coef.to_csv(RAIZ / "reports" / "coeficientes_regresion_logistica.csv", header=["coeficiente"])
print("\nCoeficientes de la regresión logística (estandarizados):")
print(coef.round(4).to_string())
