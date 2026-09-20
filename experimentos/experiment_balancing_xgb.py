"""
EXPERIMENTO COMPARATIVO - NAO E O MODELO FINAL DO PROJETO.

Compara 3 estrategias de balanceamento com o XGBoost tunado (12 variaveis
comportamentais), apenas para investigar o efeito no erro Medio->Alto risco.
O modelo oficial do trabalho continua sendo o cenario 1 (sem balanceamento),
ja avaliado nos scripts anteriores (cv_and_diagnostics_xgb.py).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
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
medio_idx = le.transform(["Medio risco"])[0]
alto_idx = le.transform(["Alto risco"])[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

XGB_PARAMS = dict(
    n_estimators=279, max_depth=3, learning_rate=0.02069721473281451,
    subsample=0.7644148053272926, colsample_bytree=0.7203513239267079,
    min_child_weight=2, gamma=0.11393619775098705,
    random_state=42, eval_metric="mlogloss", tree_method="hist",
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_t = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_t = scaler.transform(imputer.transform(X_test))

y_test_bin = label_binarize(y_test, classes=class_idx_order)


def evaluate(model, X_ev_test, cenario):
    y_pred = model.predict(X_ev_test)
    y_proba = model.predict_proba(X_ev_test)
    col_order = [list(model.classes_).index(c) for c in class_idx_order]
    y_proba_ord = y_proba[:, col_order]

    f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
    recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
    aucs = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")
    cm = confusion_matrix(y_test, y_pred, labels=class_idx_order)

    rows = []
    for i, cls in enumerate(class_order):
        rows.append({"Cenario": cenario, "Classe": cls, "AUC": aucs[i], "F1": f1s[i], "Recall": recs[i]})
    return pd.DataFrame(rows), cm


results = []
confusion_matrices = {}

# --- Cenario 1: sem balanceamento (modelo oficial, ja usado nos scripts anteriores) ---
model_1 = XGBClassifier(**XGB_PARAMS)
model_1.fit(X_train_t, y_train)
res_1, cm_1 = evaluate(model_1, X_test_t, "1. Sem balanceamento (oficial)")
results.append(res_1)
confusion_matrices["1. Sem balanceamento (oficial)"] = cm_1

# --- Cenario 2: sample_weight inversamente proporcional a frequencia da classe ---
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
model_2 = XGBClassifier(**XGB_PARAMS)
model_2.fit(X_train_t, y_train, sample_weight=sample_weights)
res_2, cm_2 = evaluate(model_2, X_test_t, "2. sample_weight balanceado")
results.append(res_2)
confusion_matrices["2. sample_weight balanceado"] = cm_2

# --- Cenario 3: SMOTE apenas no treino (apos imputacao/escala, nunca no teste) ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train)
model_3 = XGBClassifier(**XGB_PARAMS)
model_3.fit(X_train_res, y_train_res)
res_3, cm_3 = evaluate(model_3, X_test_t, "3. SMOTE (treino)")
results.append(res_3)
confusion_matrices["3. SMOTE (treino)"] = cm_3

print("Contagem de classes no treino, ANTES e DEPOIS do SMOTE:")
print("Antes:", {le.inverse_transform([c])[0]: int((y_train == c).sum()) for c in class_idx_order})
print("Depois:", {le.inverse_transform([c])[0]: int((y_train_res == c).sum()) for c in class_idx_order})

comparison = pd.concat(results, ignore_index=True)
pd.set_option("display.width", 140)
print("\n=== [EXPERIMENTO COMPARATIVO] Metricas por classe nos 3 cenarios ===")
print(comparison.round(4).to_string(index=False))

print("\n=== [EXPERIMENTO COMPARATIVO] % de Medio risco classificado como Alto risco ===")
for cenario, cm in confusion_matrices.items():
    total_medio = cm[1].sum()
    medio_como_alto = cm[1, 2]
    print(f"  {cenario}: {medio_como_alto}/{total_medio} = {medio_como_alto/total_medio*100:.1f}%")

comparison.to_csv(METRICAS / "EXPERIMENTO_comparativo_balanceamento_por_classe.csv", index=False)
print("\nSalvo em EXPERIMENTO_comparativo_balanceamento_por_classe.csv (isto e um experimento, nao o modelo final)")

# ============================================================================
# GRAFICOS - 3 matrizes de confusao lado a lado
# ============================================================================
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=200)
cenarios = list(confusion_matrices.keys())

for ax, cenario in zip(axes, cenarios):
    cm = confusion_matrices[cenario]
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(class_order, fontsize=9)
    ax.set_yticklabels(class_order, fontsize=9)
    ax.set_xlabel("Classe prevista", fontsize=9.5, color=INK_SECONDARY)
    if cenario == cenarios[0]:
        ax.set_ylabel("Classe real", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(cenario, fontsize=11, loc="left", fontweight="bold", pad=10)

    for i in range(3):
        for j in range(3):
            text_color = "white" if cm_pct[i, j] > 0.5 else INK_PRIMARY
            weight = "bold" if (i == 1 and j == 2) else "normal"  # destaca Medio->Alto
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)",
                ha="center", va="center", fontsize=9, color=text_color, fontweight=weight,
            )
            if i == 1 and j == 2:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#eb6834", linewidth=2.5)
                ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

fig.suptitle(
    "[EXPERIMENTO COMPARATIVO - nao substitui o modelo oficial]\n"
    "Matrizes de confusao por estrategia de balanceamento (destaque: Medio risco -> Alto risco)",
    fontsize=13, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold", y=1.06,
)

fig.tight_layout()
fig.savefig(FIGURAS / "EXPERIMENTO_matrizes_confusao_balanceamento.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: EXPERIMENTO_matrizes_confusao_balanceamento.png (experimento comparativo, nao e o modelo final)")


# ============================================================================
# GRAFICO - comparacao de metricas por classe (grouped bar, 3 cenarios)
# ============================================================================
SCENARIO_COLORS = {
    "1. Sem balanceamento (oficial)": "#2a78d6",
    "2. sample_weight balanceado": "#eb6834",
    "3. SMOTE (treino)": "#1baf7a",
}
METRICS = ["AUC", "F1", "Recall"]
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
INK_MUTED = "#898781"

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=200, sharey=True)
x = np.arange(len(class_order))
bar_width = 0.24

for ax, metric in zip(axes, METRICS):
    for i, cenario in enumerate(cenarios):
        vals = [
            comparison.loc[(comparison.Cenario == cenario) & (comparison.Classe == cls), metric].values[0]
            for cls in class_order
        ]
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width * 0.9,
            color=SCENARIO_COLORS[cenario], label=cenario,
            edgecolor=SURFACE, linewidth=1.5,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.8, color=INK_SECONDARY,
            )

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
    "[EXPERIMENTO COMPARATIVO - nao substitui o modelo oficial] Metricas por classe nos 3 cenarios de balanceamento",
    fontsize=12.5, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold", y=1.05,
)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(FIGURAS / "EXPERIMENTO_metricas_balanceamento.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: EXPERIMENTO_metricas_balanceamento.png (experimento comparativo, nao e o modelo final)")
