import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

df = pd.read_csv(DADOS / "bwin_dataset_agregado_classificado.csv")

# --- mesmas features/target/split do script anterior (para comparacao justa) ---
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

# scorer: recall exclusivamente da classe "Alto risco" (objetivo do tuning)
recall_alto_scorer = make_scorer(recall_score, labels=[alto_label], average="macro")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
class_idx_order = le.transform(class_order)
y_test_bin = label_binarize(y_test, classes=class_idx_order)


def evaluate(pipe, label):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    col_order = [list(pipe.classes_).index(c) for c in class_idx_order]
    y_proba_ord = y_proba[:, col_order]

    f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
    recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
    aucs = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")

    rows = []
    for i, cls in enumerate(class_order):
        rows.append({"Fase": label, "Classe": cls, "AUC": aucs[i], "F1": f1s[i], "Recall": recs[i]})
    return pd.DataFrame(rows)


results = []

# ============================== RANDOM FOREST ==============================
rf_baseline = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])
rf_baseline.fit(X_train, y_train)
results.append(evaluate(rf_baseline, "Random Forest - Antes").assign(Modelo="Random Forest"))

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
    rf_pipe, rf_param_dist, n_iter=40, scoring=recall_alto_scorer,
    cv=cv, random_state=42, n_jobs=-1, refit=True, verbose=0,
)
rf_search.fit(X_train, y_train)
print("=== Random Forest: melhores hiperparametros ===")
for k, v in rf_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"  Recall(Alto risco) medio em CV (5 folds): {rf_search.best_score_:.4f}\n")

results.append(evaluate(rf_search.best_estimator_, "Random Forest - Depois").assign(Modelo="Random Forest"))

# ================================ XGBOOST ===================================
xgb_baseline = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(n_estimators=300, random_state=42, eval_metric="mlogloss", tree_method="hist")),
])
xgb_baseline.fit(X_train, y_train)
results.append(evaluate(xgb_baseline, "XGBoost - Antes").assign(Modelo="XGBoost"))

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
print("=== XGBoost: melhores hiperparametros ===")
for k, v in xgb_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"  Recall(Alto risco) medio em CV (5 folds): {xgb_search.best_score_:.4f}\n")

results.append(evaluate(xgb_search.best_estimator_, "XGBoost - Depois").assign(Modelo="XGBoost"))

# ============================== COMPARACAO ==================================
all_results = pd.concat(results, ignore_index=True)
pd.set_option("display.width", 140)

print("=== Metricas no conjunto de teste, ANTES x DEPOIS do tuning (por classe) ===")
print(all_results[["Modelo", "Fase", "Classe", "AUC", "F1", "Recall"]].round(4).to_string(index=False))

alto_only = all_results[all_results.Classe == "Alto risco"][["Modelo", "Fase", "AUC", "F1", "Recall"]]
print("\n=== Foco: classe Alto risco (objetivo do tuning) ===")
print(alto_only.round(4).to_string(index=False))

all_results.to_csv(METRICAS / "tuning_before_after_by_class.csv", index=False)
print("\nSalvo em tuning_before_after_by_class.csv")
