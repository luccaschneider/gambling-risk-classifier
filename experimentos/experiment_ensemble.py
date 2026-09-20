"""
EXPERIMENTO - ensemble do Random Forest tunado + XGBoost tunado (12 variaveis
comportamentais). Compara hard voting (maioria) e soft voting (media das
probabilidades) com os dois modelos individuais.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

RF_PARAMS = dict(
    n_estimators=529, max_depth=10, max_features="sqrt",
    min_samples_leaf=2, min_samples_split=11, class_weight="balanced",
    random_state=42, n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=279, max_depth=3, learning_rate=0.02069721473281451,
    subsample=0.7644148053272926, colsample_bytree=0.7203513239267079,
    min_child_weight=2, gamma=0.11393619775098705,
    random_state=42, eval_metric="mlogloss", tree_method="hist",
)


def make_pipe(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


rf_pipe = make_pipe(RandomForestClassifier(**RF_PARAMS))
xgb_pipe = make_pipe(XGBClassifier(**XGB_PARAMS))
rf_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)

y_test_bin = label_binarize(y_test, classes=class_idx_order)


def ordered_proba(pipe, X_ev):
    proba = pipe.predict_proba(X_ev)
    col_order = [list(pipe.named_steps["clf"].classes_).index(c) for c in class_idx_order]
    return proba[:, col_order]


proba_rf = ordered_proba(rf_pipe, X_test)
proba_xgb = ordered_proba(xgb_pipe, X_test)
pred_rf = class_idx_order[np.argmax(proba_rf, axis=1)]
pred_xgb = class_idx_order[np.argmax(proba_xgb, axis=1)]

# --- SOFT VOTING: media das probabilidades -----------------------------------
proba_soft = (proba_rf + proba_xgb) / 2
pred_soft = class_idx_order[np.argmax(proba_soft, axis=1)]

# --- HARD VOTING: maioria simples ---------------------------------------------
# com apenas 2 modelos-base, "maioria" so existe quando os dois concordam;
# quando discordam (empate 1-1), o desempate usa a media das probabilidades
# (equivale a cair no soft voting so nos casos de empate)
agree_mask = pred_rf == pred_xgb
n_disagree = (~agree_mask).sum()
pred_hard = np.where(agree_mask, pred_rf, pred_soft)

print(f"Casos em que RF e XGBoost discordam (empate 1-1, desempate por probabilidade media): "
      f"{n_disagree}/{len(y_test)} ({n_disagree/len(y_test)*100:.1f}%)")


def evaluate(y_pred, proba_ord, modelo, has_proba=True):
    f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
    recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
    rows = []
    if has_proba:
        aucs = roc_auc_score(y_test_bin, proba_ord, average=None, multi_class="ovr")
    else:
        aucs = [np.nan] * 3
    for i, cls in enumerate(class_order):
        rows.append({"Modelo": modelo, "Classe": cls, "AUC": aucs[i], "F1": f1s[i], "Recall": recs[i]})
    return pd.DataFrame(rows)


results = []
results.append(evaluate(pred_rf, proba_rf, "Random Forest (individual)"))
results.append(evaluate(pred_xgb, proba_xgb, "XGBoost (individual)"))
# hard voting: AUC nao e bem definida (voto binario, nao probabilistico) - reportada como NaN
results.append(evaluate(pred_hard, None, "Ensemble - Hard voting", has_proba=False))
results.append(evaluate(pred_soft, proba_soft, "Ensemble - Soft voting"))

comparison = pd.concat(results, ignore_index=True)
pd.set_option("display.width", 140)
print("\n=== [EXPERIMENTO] Metricas por classe: individuais x ensembles ===")
print(comparison.round(4).to_string(index=False))

summary_rows = []
for modelo in comparison.Modelo.unique():
    sub = comparison[comparison.Modelo == modelo]
    summary_rows.append({
        "Modelo": modelo,
        "AUC_macro": sub["AUC"].mean(),
        "F1_macro": sub["F1"].mean(),
        "Recall_macro": sub["Recall"].mean(),
    })
summary = pd.DataFrame(summary_rows).sort_values("F1_macro", ascending=False)
print("\n=== [EXPERIMENTO] Resumo macro ===")
print(summary.round(4).to_string(index=False))

comparison.to_csv(METRICAS / "EXPERIMENTO_ensemble_por_classe.csv", index=False)
summary.to_csv(METRICAS / "EXPERIMENTO_ensemble_resumo_macro.csv", index=False)
print("\nSalvo em EXPERIMENTO_ensemble_por_classe.csv e EXPERIMENTO_ensemble_resumo_macro.csv")

# melhor ensemble = maior F1 macro entre os dois modos de votacao
ensemble_summary = summary[summary.Modelo.str.startswith("Ensemble")]
best_ensemble_name = ensemble_summary.iloc[0]["Modelo"]
best_pred = pred_soft if "Soft" in best_ensemble_name else pred_hard
print(f"\nMelhor ensemble (por F1 macro): {best_ensemble_name}")

# ============================================================================
# GRAFICOS
# ============================================================================
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
})

MODEL_COLORS = {
    "Random Forest (individual)": "#2a78d6",
    "XGBoost (individual)": "#eb6834",
    "Ensemble - Hard voting": "#1baf7a",
    "Ensemble - Soft voting": "#eda100",
}
METRICS = ["AUC", "F1", "Recall"]
model_order = ["Random Forest (individual)", "XGBoost (individual)", "Ensemble - Hard voting", "Ensemble - Soft voting"]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=200, sharey=True)
x = np.arange(len(class_order))
bar_width = 0.2

for ax, metric in zip(axes, METRICS):
    for i, modelo in enumerate(model_order):
        vals = [
            comparison.loc[(comparison.Modelo == modelo) & (comparison.Classe == cls), metric].values[0]
            for cls in class_order
        ]
        offset = (i - 1.5) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width * 0.9,
            color=MODEL_COLORS[modelo], label=modelo,
            edgecolor=SURFACE, linewidth=1.2,
        )
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(rect.get_x() + rect.get_width() / 2, 0.02, "N/A",
                        ha="center", va="bottom", fontsize=6.5, color=INK_MUTED, rotation=90)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.3, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(class_order, fontsize=9.5, rotation=8)
    ax.set_ylim(0, 1.12)
    ax.set_title(metric, fontsize=12, loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)

fig.suptitle(
    "[EXPERIMENTO] Individuais x ensembles (RF + XGBoost tunados, 12 var. comportamentais)",
    fontsize=13, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold", y=1.06,
)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(FIGURAS / "EXPERIMENTO_ensemble_metricas.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: EXPERIMENTO_ensemble_metricas.png")

# --- matriz de confusao do melhor ensemble -----------------------------------
cm = confusion_matrix(y_test, best_pred, labels=class_idx_order)
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
    f"[EXPERIMENTO] Matriz de confusao - melhor ensemble\n({best_ensemble_name})",
    fontsize=12.5, loc="left", fontweight="bold", pad=14,
)
for i in range(3):
    for j in range(3):
        text_color = "white" if cm_pct[i, j] > 0.5 else INK_PRIMARY
        ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)", ha="center", va="center",
                 fontsize=10, color=text_color)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8.5, length=0)
cbar.outline.set_visible(False)
cbar.set_label("Proporcao dentro da classe real", fontsize=9, color=INK_SECONDARY)

fig.tight_layout()
fig.savefig(FIGURAS / "EXPERIMENTO_ensemble_matriz_confusao.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: EXPERIMENTO_ensemble_matriz_confusao.png")
