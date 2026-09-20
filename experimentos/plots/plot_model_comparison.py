import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[2]
METRICAS = RAIZ / "resultados" / "metricas"
FIGURAS = RAIZ / "resultados" / "figuras"

detail = pd.read_csv(METRICAS / "model_comparison_by_class.csv")
summary = pd.read_csv(METRICAS / "model_comparison_summary.csv")

MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
CLASS_ORDER = ["Baixo risco", "Medio risco", "Alto risco"]
COLORS = {
    "Logistic Regression": "#2a78d6",  # blue
    "Decision Tree": "#eb6834",        # orange
    "Random Forest": "#1baf7a",        # aqua
    "XGBoost": "#eda100",              # yellow
}

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


def grouped_bar_by_class(metric, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)

    n_models = len(MODEL_ORDER)
    n_classes = len(CLASS_ORDER)
    group_width = 0.72
    bar_width = group_width / n_models
    x = np.arange(n_classes)

    for i, model in enumerate(MODEL_ORDER):
        vals = [
            detail.loc[(detail.Modelo == model) & (detail.Classe == cls), metric].values[0]
            for cls in CLASS_ORDER
        ]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width * 0.88,
            color=COLORS[model], label=model,
            edgecolor=SURFACE, linewidth=2,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, fontsize=10.5, color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=13, color=INK_PRIMARY, pad=14, loc="left", fontweight="bold")

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
        frameon=False, fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {filename}")


grouped_bar_by_class("AUC", "AUC por modelo e classe de risco", FIGURAS / "grafico_auc_por_classe.png")
grouped_bar_by_class("F1", "F1-score por modelo e classe de risco", FIGURAS / "grafico_f1_por_classe.png")
grouped_bar_by_class("Recall", "Recall por modelo e classe de risco", FIGURAS / "grafico_recall_por_classe.png")


# --- resumo macro (visao geral por modelo) ---------------------------------
def macro_summary_chart(filename):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)

    metrics = ["Acuracia", "AUC_macro", "F1_macro", "Recall_macro"]
    metric_labels = ["Acurácia", "AUC (macro)", "F1 (macro)", "Recall (macro)"]
    n_models = len(MODEL_ORDER)
    n_metrics = len(metrics)
    group_width = 0.72
    bar_width = group_width / n_models
    x = np.arange(n_metrics)

    for i, model in enumerate(MODEL_ORDER):
        vals = [summary.loc[summary.Modelo == model, m].values[0] for m in metrics]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, width=bar_width * 0.88,
            color=COLORS[model], label=model,
            edgecolor=SURFACE, linewidth=2,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5, color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Resumo comparativo dos modelos (médias macro no teste)",
        fontsize=13, color=INK_PRIMARY, pad=14, loc="left", fontweight="bold",
    )

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
        frameon=False, fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {filename}")


macro_summary_chart(FIGURAS / "grafico_resumo_macro.png")
