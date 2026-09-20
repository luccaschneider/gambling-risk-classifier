import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

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

preprocess = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_test_bin = label_binarize(y_test, classes=class_idx_order)


def evaluate(pipe, label, modelo):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    col_order = [list(pipe.classes_).index(c) for c in class_idx_order]
    y_proba_ord = y_proba[:, col_order]

    f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
    recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
    aucs = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")

    rows = []
    for i, cls in enumerate(class_order):
        rows.append({"Modelo": modelo, "Fase": label, "Classe": cls, "AUC": aucs[i], "F1": f1s[i], "Recall": recs[i]})
    return pd.DataFrame(rows)


results = []

# ============================== RANDOM FOREST ==============================
rf_baseline = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])
rf_baseline.fit(X_train, y_train)
results.append(evaluate(rf_baseline, "Antes", "Random Forest"))

rf_param_dist = {
    "clf__n_estimators": randint(150, 600),
    "clf__max_depth": [None, 5, 10, 15, 20, 30],
    "clf__min_samples_split": randint(2, 12),
    "clf__min_samples_leaf": randint(1, 6),
    "clf__max_features": ["sqrt", "log2", None],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}
rf_pipe = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))])
rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, n_iter=40, scoring="recall_macro",
    cv=cv, random_state=42, n_jobs=-1, refit=True, verbose=0,
)
rf_search.fit(X_train, y_train)
print("=== Random Forest: melhores hiperparametros (otimizando recall macro) ===")
for k, v in rf_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"  Recall macro em CV (5 folds): {rf_search.best_score_:.4f}\n")

results.append(evaluate(rf_search.best_estimator_, "Depois", "Random Forest"))

# ================================ XGBOOST ===================================
xgb_baseline = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(n_estimators=300, random_state=42, eval_metric="mlogloss", tree_method="hist")),
])
xgb_baseline.fit(X_train, y_train)
results.append(evaluate(xgb_baseline, "Antes", "XGBoost"))

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
    xgb_pipe, xgb_param_dist, n_iter=40, scoring="recall_macro",
    cv=cv, random_state=42, n_jobs=-1, refit=True, verbose=0,
)
xgb_search.fit(X_train, y_train)
print("=== XGBoost: melhores hiperparametros (otimizando recall macro) ===")
for k, v in xgb_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"  Recall macro em CV (5 folds): {xgb_search.best_score_:.4f}\n")

results.append(evaluate(xgb_search.best_estimator_, "Depois", "XGBoost"))

# ============================== COMPARACAO ==================================
all_results = pd.concat(results, ignore_index=True)
pd.set_option("display.width", 140)

print("=== Metricas no teste, ANTES x DEPOIS do tuning (por classe, 12 variaveis comportamentais) ===")
print(all_results[["Modelo", "Fase", "Classe", "AUC", "F1", "Recall"]].round(4).to_string(index=False))

summary_rows = []
for modelo in ["Random Forest", "XGBoost"]:
    for fase in ["Antes", "Depois"]:
        sub = all_results[(all_results.Modelo == modelo) & (all_results.Fase == fase)]
        summary_rows.append({
            "Modelo": modelo, "Fase": fase,
            "AUC_macro": sub["AUC"].mean(), "F1_macro": sub["F1"].mean(), "Recall_macro": sub["Recall"].mean(),
        })
summary = pd.DataFrame(summary_rows)
print("\n=== Resumo macro (media das 3 classes) ===")
print(summary.round(4).to_string(index=False))

all_results.to_csv(METRICAS / "tuning_behavioral_before_after_by_class.csv", index=False)
summary.to_csv(METRICAS / "tuning_behavioral_before_after_summary.csv", index=False)
print("\nSalvo em tuning_behavioral_before_after_by_class.csv e ..._summary.csv")
