import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, confusion_matrix, f1_score, recall_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"
FIGURAS = RAIZ / "resultados" / "figuras"

df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")

behavioral_features = [
    "total_dias_ativos", "total_apostado", "total_perdas", "numero_total_apostas",
    "variedade_produtos", "media_diaria_apostada", "desvio_padrao_apostado",
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
]

X = df[behavioral_features]
y = df["classificacao_risco"]

class_order = ["Baixo risco", "Medio risco", "Alto risco"]
le = LabelEncoder()
le.fit(class_order)
y_enc = le.transform(y)
class_idx_order = le.transform(class_order)

# hiperparametros do XGBoost tunado (RandomizedSearchCV, recall macro, 12 var. comportamentais)
XGB_PARAMS = dict(
    n_estimators=279, max_depth=3, learning_rate=0.02069721473281451,
    subsample=0.7644148053272926, colsample_bytree=0.7203513239267079,
    min_child_weight=2, gamma=0.11393619775098705,
    random_state=42, eval_metric="mlogloss", tree_method="hist",
)


def make_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", XGBClassifier(**XGB_PARAMS)),
    ])


# ============================================================================
# 1) VALIDACAO CRUZADA ESTRATIFICADA - 10 FOLDS (dataset completo)
# ============================================================================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_rows = []

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

    pipe = make_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)

    col_order = [list(pipe.classes_).index(c) for c in class_idx_order]
    y_proba_ord = y_proba[:, col_order]
    y_val_bin = label_binarize(y_val, classes=class_idx_order)

    f1s = f1_score(y_val, y_pred, labels=class_idx_order, average=None)
    recs = recall_score(y_val, y_pred, labels=class_idx_order, average=None)
    aucs = roc_auc_score(y_val_bin, y_proba_ord, average=None, multi_class="ovr")

    row = {"fold": fold_i}
    for i, cls in enumerate(class_order):
        row[f"AUC_{cls}"] = aucs[i]
        row[f"F1_{cls}"] = f1s[i]
        row[f"Recall_{cls}"] = recs[i]
    row["AUC_macro"] = aucs.mean()
    row["F1_macro"] = f1s.mean()
    row["Recall_macro"] = recs.mean()
    fold_rows.append(row)

cv_df = pd.DataFrame(fold_rows)
cv_df.to_csv(METRICAS / "cv10_xgb_behavioral_folds.csv", index=False)

metric_cols = [c for c in cv_df.columns if c != "fold"]
cv_summary = pd.DataFrame({
    "media": cv_df[metric_cols].mean(),
    "desvio_padrao": cv_df[metric_cols].std(),
})
cv_summary.to_csv(METRICAS / "cv10_xgb_behavioral_summary.csv")

pd.set_option("display.width", 140)
print("=== Validacao cruzada estratificada (10 folds) - XGBoost tunado, 12 var. comportamentais ===")
print("\n-- Macro --")
print(cv_summary.loc[["AUC_macro", "F1_macro", "Recall_macro"]].round(4).to_string())
print("\n-- Por classe --")
for cls in class_order:
    print(f"\n{cls}:")
    print(cv_summary.loc[[f"AUC_{cls}", f"F1_{cls}", f"Recall_{cls}"]].round(4).to_string())

# ============================================================================
# 2) MODELO FINAL NO SPLIT 80/20 -> MATRIZ DE CONFUSAO E CURVA ROC
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

final_pipe = make_pipeline()
final_pipe.fit(X_train, y_train)
y_pred_test = final_pipe.predict(X_test)
y_proba_test = final_pipe.predict_proba(X_test)
col_order = [list(final_pipe.classes_).index(c) for c in class_idx_order]
y_proba_test_ord = y_proba_test[:, col_order]
y_test_bin = label_binarize(y_test, classes=class_idx_order)

# --- estilo consistente com os graficos anteriores --------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
RISK_COLORS = {"Baixo risco": "#1baf7a", "Medio risco": "#eda100", "Alto risco": "#e34948"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
})

# --- matriz de confusao (heatmap sequencial azul) ----------------------------
cm = confusion_matrix(y_test, y_pred_test, labels=class_idx_order)
cm_pct = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(class_order, fontsize=10)
ax.set_yticklabels(class_order, fontsize=10)
ax.set_xlabel("Classe prevista", fontsize=10.5, color=INK_SECONDARY)
ax.set_ylabel("Classe real", fontsize=10.5, color=INK_SECONDARY)
ax.set_title(
    "Matriz de confusao - XGBoost tunado (teste, 20%)",
    fontsize=13, loc="left", fontweight="bold", pad=14,
)

for i in range(3):
    for j in range(3):
        text_color = "white" if cm_pct[i, j] > 0.5 else INK_PRIMARY
        ax.text(
            j, i, f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)",
            ha="center", va="center", fontsize=10, color=text_color,
        )

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8.5, length=0)
cbar.outline.set_visible(False)
cbar.set_label("Proporcao dentro da classe real", fontsize=9, color=INK_SECONDARY)

fig.tight_layout()
fig.savefig(FIGURAS / "matriz_confusao_xgb_tunado.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSalvo: matriz_confusao_xgb_tunado.png")

# --- curva ROC (one-vs-rest, 3 classes) --------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=200)

for i, cls in enumerate(class_order):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_test_ord[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr, tpr, color=RISK_COLORS[cls], linewidth=2.2,
        label=f"{cls} (AUC = {roc_auc:.3f})",
    )

ax.plot([0, 1], [0, 1], linestyle="--", color=BASELINE, linewidth=1.4, label="Aleatorio (AUC = 0.500)")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Taxa de falsos positivos", fontsize=10.5, color=INK_SECONDARY)
ax.set_ylabel("Taxa de verdadeiros positivos (recall)", fontsize=10.5, color=INK_SECONDARY)
ax.set_title(
    "Curva ROC (one-vs-rest) - XGBoost tunado (teste, 20%)",
    fontsize=13, loc="left", fontweight="bold", pad=14,
)
ax.grid(color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BASELINE)
ax.tick_params(axis="both", length=0, labelsize=9.5, colors=INK_MUTED)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(FIGURAS / "curva_roc_xgb_tunado.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: curva_roc_xgb_tunado.png")
