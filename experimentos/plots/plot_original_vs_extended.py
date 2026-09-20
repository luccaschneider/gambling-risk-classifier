import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[2]
METRICAS = RAIZ / "resultados" / "metricas"
FIGURAS = RAIZ / "resultados" / "figuras"

df = pd.read_csv(METRICAS / "comparison_original_vs_extended_behavioral.csv")
importances = pd.read_csv(METRICAS / "feature_importance_extended_rf.csv", index_col=0)

CLASS_ORDER = ["Baixo risco", "Medio risco", "Alto risco"]
METRICS = ["AUC", "F1", "Recall"]
FEATURE_COLORS = {"Comportamentais originais": "#2a78d6", "Comportamentais + novas": "#1baf7a"}  # blue / aqua

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEW_FEATURES = {
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


def comparison_chart(model_name, filename):
    sub = df[df.Modelo == model_name].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200, sharey=True)
    x = np.arange(len(CLASS_ORDER))
    bar_width = 0.32

    for ax, metric in zip(axes, METRICS):
        for i, feat in enumerate(["Comportamentais originais", "Comportamentais + novas"]):
            vals = [
                sub.loc[(sub.Features == feat) & (sub.Classe == cls), metric].values[0]
                for cls in CLASS_ORDER
            ]
            offset = (i - 0.5) * bar_width
            bars = ax.bar(
                x + offset, vals, width=bar_width * 0.9,
                color=FEATURE_COLORS[feat], label=feat,
                edgecolor=SURFACE, linewidth=2,
            )
            for rect, v in zip(bars, vals):
                ax.text(
                    rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_ORDER, fontsize=9.5, color=INK_PRIMARY, rotation=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(metric, fontsize=12, color=INK_PRIMARY, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="both", length=0)

    fig.suptitle(
        f"{model_name}: variaveis comportamentais originais x com novas variaveis derivadas",
        fontsize=13.5, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold", y=1.04,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
        frameon=False, fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {filename}")


comparison_chart("Random Forest", FIGURAS / "grafico_original_vs_estendido_rf.png")
comparison_chart("XGBoost", FIGURAS / "grafico_original_vs_estendido_xgb.png")


# --- importancia das variaveis, destacando as novas -------------------------
imp_sorted = importances["importancia"].sort_values(ascending=True)
colors = ["#1baf7a" if f in NEW_FEATURES else "#2a78d6" for f in imp_sorted.index]

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=200)
y_pos = np.arange(len(imp_sorted))
ax.barh(y_pos, imp_sorted.values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(imp_sorted.index, fontsize=9.5)
ax.set_xlabel("Importancia (Gini) - Random Forest", fontsize=10, color=INK_SECONDARY)
ax.set_title(
    "Importancia das variaveis comportamentais (originais x novas)",
    fontsize=13, loc="left", fontweight="bold", pad=14,
)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="both", length=0)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#2a78d6", label="Variaveis originais"),
    Patch(facecolor="#1baf7a", label="Novas variaveis derivadas"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(FIGURAS / "grafico_importancia_novas_variaveis.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: grafico_importancia_novas_variaveis.png")
