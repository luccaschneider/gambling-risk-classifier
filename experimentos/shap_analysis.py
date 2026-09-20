import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"
FIGURAS = RAIZ / "resultados" / "figuras"

df = pd.read_csv(DADOS / "bwin_dataset_agregado_classificado.csv")

df["YearofBirth"] = pd.to_numeric(df["YearofBirth"], errors="coerce")
df["Registration_date"] = pd.to_datetime(df["Registration_date"], errors="coerce")
df["First_Deposit_Date"] = pd.to_datetime(df["First_Deposit_Date"], errors="coerce")
df["dias_ate_primeiro_deposito"] = (df["First_Deposit_Date"] - df["Registration_date"]).dt.days
df["ano_registro"] = df["Registration_date"].dt.year

numeric_features = [
    "total_dias_ativos", "total_apostado", "total_perdas", "numero_total_apostas",
    "variedade_produtos", "media_diaria_apostada", "desvio_padrao_apostado",
    "YearofBirth", "dias_ate_primeiro_deposito", "ano_registro",
]
categorical_features = ["CountryName", "LanguageName", "Gender"]

X = df[numeric_features + categorical_features]
y = df["classificacao_risco"]

class_order = ["Baixo risco", "Medio risco", "Alto risco"]
le = LabelEncoder()
le.fit(class_order)
y_enc = le.transform(y)
alto_label = le.transform(["Alto risco"])[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

recall_alto_scorer = make_scorer(recall_score, labels=[alto_label], average="macro")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_param_dist = {
    "clf__n_estimators": randint(150, 600),
    "clf__max_depth": randint(3, 10),
    "clf__learning_rate": uniform(0.01, 0.29),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 8),
    "clf__gamma": uniform(0, 0.4),
}
xgb_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(random_state=42, eval_metric="mlogloss", tree_method="hist")),
])
xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_param_dist, n_iter=40, scoring=recall_alto_scorer,
    cv=cv, random_state=42, n_jobs=-1, refit=True, verbose=0,
)
xgb_search.fit(X_train, y_train)
best_pipe = xgb_search.best_estimator_
prep = best_pipe.named_steps["prep"]
model = best_pipe.named_steps["clf"]

# --- prepara dados transformados p/ SHAP -----------------------------------
feature_names = prep.get_feature_names_out()
feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

X_test_t = prep.transform(X_test)
if hasattr(X_test_t, "toarray"):
    X_test_t = X_test_t.toarray()
X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_df)  # Explanation, values shape (n, n_features, n_classes)

class_names_sorted = list(le.inverse_transform(sorted(set(y_enc))))
alto_idx_in_model = list(model.classes_).index(alto_label)

# --- estilo consistente com os graficos anteriores --------------------------
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

# 1) Importancia global (media |SHAP| por classe, top 15 variaveis) ----------
mean_abs_per_class = np.abs(shap_values.values).mean(axis=0)  # (n_features, n_classes)
mean_abs_overall = mean_abs_per_class.mean(axis=1)
top_idx = np.argsort(mean_abs_overall)[::-1][:15]

risk_colors = {"Baixo risco": "#1baf7a", "Medio risco": "#eda100", "Alto risco": "#e34948"}  # verde/amarelo/vermelho
fig, ax = plt.subplots(figsize=(9.5, 7), dpi=200)
y_pos = np.arange(len(top_idx))
left = np.zeros(len(top_idx))
model_classes_sorted = sorted(set(y_enc))
for cls_name in class_order:  # Baixo, Medio, Alto - ordem semantica de risco
    c_i = model_classes_sorted.index(le.transform([cls_name])[0])
    vals = mean_abs_per_class[top_idx, c_i]
    ax.barh(
        y_pos, vals, left=left, height=0.62,
        color=risk_colors[cls_name], label=cls_name, edgecolor=SURFACE, linewidth=1.5,
    )
    left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Media de |valor SHAP| (impacto no modelo)", fontsize=10, color=INK_SECONDARY)
ax.set_title(
    "Importancia das variaveis - XGBoost tunado (SHAP)",
    fontsize=13, loc="left", fontweight="bold", pad=14,
)
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="both", length=0)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(FIGURAS / "shap_importancia_geral.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: shap_importancia_geral.png")

# 2) Beeswarm para a classe Alto risco (top 15) -------------------------------
shap_alto = shap_values[:, :, alto_idx_in_model]
fig = plt.figure(figsize=(9.5, 7), dpi=200)
shap.plots.beeswarm(shap_alto, max_display=15, show=False)
fig = plt.gcf()
fig.set_facecolor(SURFACE)
for ax in fig.axes:
    ax.set_facecolor(SURFACE)
plt.title(
    "SHAP - impacto das variaveis na predicao de Alto risco (XGBoost tunado)",
    fontsize=12, loc="left", fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURAS / "shap_beeswarm_alto_risco.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvo: shap_beeswarm_alto_risco.png")

# ranking impresso p/ referencia
ranking = pd.DataFrame({
    "variavel": [feature_names[i] for i in top_idx],
    "shap_medio_abs": mean_abs_overall[top_idx],
})
print("\nTop 15 variaveis por importancia SHAP (media entre as 3 classes):")
print(ranking.round(4).to_string(index=False))
ranking.to_csv(METRICAS / "shap_ranking_variaveis.csv", index=False)
