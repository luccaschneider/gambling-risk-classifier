import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[2]
METRICAS = RAIZ / "resultados" / "metricas"
FIGURAS = RAIZ / "resultados" / "figuras"

df = pd.read_csv(METRICAS / "tuning_before_after_by_class.csv")

CLASS_ORDER = ["Baixo risco", "Medio risco", "Alto risco"]
METRICS = ["AUC", "F1", "Recall"]

PHASE_COLORS = {"Antes": "#2a78d6", "Depois": "#eb6834"}  # slot1 blue / slot2 orange

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
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


def tuning_chart(model_name, filename):
    sub = df[df.Modelo == model_name].copy()
    sub["Fase"] = sub["Fase"].str.split(" - ").str[1]  # "Antes" / "Depois"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200, sharey=True)

    x = np.arange(len(CLASS_ORDER))
    bar_width = 0.32

    for ax, metric in zip(axes, METRICS):
        for i, phase in enumerate(["Antes", "Depois"]):
            vals = [
                sub.loc[(sub.Fase == phase) & (sub.Classe == cls), metric].values[0]
                for cls in CLASS_ORDER
            ]
            offset = (i - 0.5) * bar_width
            bars = ax.bar(
                x + offset, vals, width=bar_width * 0.9,
                color=PHASE_COLORS[phase], label=phase,
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
        f"{model_name}: antes x depois do tuning (foco: recall de Alto risco)",
        fontsize=14, color=INK_PRIMARY, x=0.02, ha="left", fontweight="bold", y=1.04,
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


tuning_chart("Random Forest", FIGURAS / "grafico_tuning_random_forest.png")
tuning_chart("XGBoost", FIGURAS / "grafico_tuning_xgboost.png")
