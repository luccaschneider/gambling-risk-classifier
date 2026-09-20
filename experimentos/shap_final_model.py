"""
Analise SHAP do modelo OFICIAL final: XGBoost tunado, 12 variaveis
comportamentais, sem balanceamento (mesmo modelo de cv_and_diagnostics_xgb.py).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

XGB_PARAMS = dict(
    n_estimators=279, max_depth=3, learning_rate=0.02069721473281451,
    subsample=0.7644148053272926, colsample_bytree=0.7203513239267079,
    min_child_weight=2, gamma=0.11393619775098705,
    random_state=42, eval_metric="mlogloss", tree_method="hist",
)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", XGBClassifier(**XGB_PARAMS)),
])
pipe.fit(X_train, y_train)
model = pipe.named_steps["clf"]

X_test_t = pipe[:-1].transform(X_test)
X_test_df = pd.DataFrame(X_test_t, columns=behavioral_features)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_df)  # shape (n, n_features, n_classes)

model_classes_sorted = sorted(set(y_enc))
baixo_label = le.transform(["Baixo risco"])[0]
alto_label = le.transform(["Alto risco"])[0]
medio_label = le.transform(["Medio risco"])[0]
baixo_idx = model_classes_sorted.index(baixo_label)
alto_idx = model_classes_sorted.index(alto_label)
medio_idx = model_classes_sorted.index(medio_label)

# --- estilo consistente com os graficos anteriores --------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
RISK_COLORS = {"Baixo risco": "#1baf7a", "Medio risco": "#eda100", "Alto risco": "#e34948"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
})

# ============================================================================
# 1) Importancia geral - todas as 12 variaveis, empilhado por classe
# ============================================================================
mean_abs_per_class = np.abs(shap_values.values).mean(axis=0)  # (n_features, n_classes)
mean_abs_overall = mean_abs_per_class.mean(axis=1)
order_idx = np.argsort(mean_abs_overall)[::-1]

fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=200)
y_pos = np.arange(len(order_idx))
left = np.zeros(len(order_idx))
for cls_name in class_order:
    c_i = model_classes_sorted.index(le.transform([cls_name])[0])
    vals = mean_abs_per_class[order_idx, c_i]
    ax.barh(
        y_pos, vals, left=left, height=0.62,
        color=RISK_COLORS[cls_name], label=cls_name, edgecolor=SURFACE, linewidth=1.5,
    )
    left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels([behavioral_features[i] for i in order_idx], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Media de |valor SHAP| (impacto no modelo)", fontsize=10, color=INK_SECONDARY)
ax.set_title(
    "Importancia das variaveis - modelo oficial (XGBoost tunado, 12 var. comportamentais)",
    fontsize=12.5, loc="left", fontweight="bold", pad=14,
)
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="both", length=0)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(FIGURAS / "shap_final_importancia_geral.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: shap_final_importancia_geral.png")

# ============================================================================
# 2) Beeswarm - Alto risco
# ============================================================================
shap_alto = shap_values[:, :, alto_idx]
fig = plt.figure(figsize=(9, 6.5), dpi=200)
shap.plots.beeswarm(shap_alto, max_display=12, show=False)
fig = plt.gcf()
fig.set_facecolor(SURFACE)
for ax in fig.axes:
    ax.set_facecolor(SURFACE)
plt.title(
    "SHAP - impacto das variaveis na predicao de Alto risco (modelo oficial)",
    fontsize=12, loc="left", fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURAS / "shap_final_beeswarm_alto_risco.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: shap_final_beeswarm_alto_risco.png")

# ============================================================================
# 3) Beeswarm - Medio risco
# ============================================================================
shap_medio = shap_values[:, :, medio_idx]
fig = plt.figure(figsize=(9, 6.5), dpi=200)
shap.plots.beeswarm(shap_medio, max_display=12, show=False)
fig = plt.gcf()
fig.set_facecolor(SURFACE)
for ax in fig.axes:
    ax.set_facecolor(SURFACE)
plt.title(
    "SHAP - impacto das variaveis na predicao de Medio risco (modelo oficial)",
    fontsize=12, loc="left", fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURAS / "shap_final_beeswarm_medio_risco.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: shap_final_beeswarm_medio_risco.png")

# ============================================================================
# 3b) Beeswarm - Baixo risco
# ============================================================================
shap_baixo = shap_values[:, :, baixo_idx]
fig = plt.figure(figsize=(9, 6.5), dpi=200)
shap.plots.beeswarm(shap_baixo, max_display=12, show=False)
fig = plt.gcf()
fig.set_facecolor(SURFACE)
for ax in fig.axes:
    ax.set_facecolor(SURFACE)
plt.title(
    "SHAP - impacto das variaveis na predicao de Baixo risco (modelo oficial)",
    fontsize=12, loc="left", fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURAS / "shap_final_beeswarm_baixo_risco.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: shap_final_beeswarm_baixo_risco.png")

# ============================================================================
# 4) Ranking numerico completo (CSV)
# ============================================================================
ranking = pd.DataFrame({
    "variavel": [behavioral_features[i] for i in order_idx],
    "shap_medio_abs_geral": mean_abs_overall[order_idx],
    "shap_medio_abs_baixo_risco": mean_abs_per_class[order_idx, model_classes_sorted.index(le.transform(["Baixo risco"])[0])],
    "shap_medio_abs_medio_risco": mean_abs_per_class[order_idx, medio_idx],
    "shap_medio_abs_alto_risco": mean_abs_per_class[order_idx, alto_idx],
})
print("\nRanking completo (12 variaveis) por importancia SHAP:")
print(ranking.round(4).to_string(index=False))

ranking.to_csv(METRICAS / "shap_final_ranking_completo.csv", index=False)
print("\nSalvo: shap_final_ranking_completo.csv")
