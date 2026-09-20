import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")

original_features = [
    "total_dias_ativos", "total_apostado", "total_perdas", "numero_total_apostas",
    "variedade_produtos", "media_diaria_apostada", "desvio_padrao_apostado",
]
new_features = [
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
]
extended_features = original_features + new_features

X = df[extended_features]
y = df["classificacao_risco"]

class_order = ["Baixo risco", "Medio risco", "Alto risco"]
le = LabelEncoder()
le.fit(class_order)
y_enc = le.transform(y)
class_idx_order = le.transform(class_order)

# mesmo split dos scripts anteriores (mesmo conjunto de teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

preprocess = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

y_test_bin = label_binarize(y_test, classes=class_idx_order)


def evaluate(pipe, modelo):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    col_order = [list(pipe.classes_).index(c) for c in class_idx_order]
    y_proba_ord = y_proba[:, col_order]

    f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
    recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
    aucs = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")

    rows = []
    for i, cls in enumerate(class_order):
        rows.append({"Modelo": modelo, "Classe": cls, "AUC": aucs[i], "F1": f1s[i], "Recall": recs[i]})
    return pd.DataFrame(rows)


rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)
rf_result = evaluate(rf_pipe, "Random Forest")

xgb_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(n_estimators=300, random_state=42, eval_metric="mlogloss", tree_method="hist")),
])
xgb_pipe.fit(X_train, y_train)
xgb_result = evaluate(xgb_pipe, "XGBoost")

extended = pd.concat([rf_result, xgb_result], ignore_index=True)
extended["Features"] = "Comportamentais + novas"

# resultados anteriores (apenas as 7 variaveis comportamentais originais)
previous = pd.read_csv(METRICAS / "comparison_demographics_vs_behavioral_only.csv")
previous = previous[previous.Features == "Apenas comportamentais"].copy()
previous["Features"] = "Comportamentais originais"

comparison = pd.concat([previous, extended], ignore_index=True)
comparison = comparison[["Modelo", "Features", "Classe", "AUC", "F1", "Recall"]]

pd.set_option("display.width", 140)
print("=== Comparacao: comportamentais originais x comportamentais + novas variaveis ===")
print(comparison.round(4).to_string(index=False))

summary_rows = []
for modelo in ["Random Forest", "XGBoost"]:
    for feat in ["Comportamentais originais", "Comportamentais + novas"]:
        sub = comparison[(comparison.Modelo == modelo) & (comparison.Features == feat)]
        summary_rows.append({
            "Modelo": modelo, "Features": feat,
            "AUC_macro": sub["AUC"].mean(), "F1_macro": sub["F1"].mean(), "Recall_macro": sub["Recall"].mean(),
        })
summary = pd.DataFrame(summary_rows)
print("\n=== Resumo macro (media das 3 classes) ===")
print(summary.round(4).to_string(index=False))

# importancia das variaveis (Random Forest) para ver onde as novas features ficam no ranking
importances = pd.Series(
    rf_pipe.named_steps["clf"].feature_importances_, index=extended_features
).sort_values(ascending=False)
print("\n=== Importancia das variaveis - Random Forest (Gini) ===")
print(importances.round(4).to_string())

comparison.to_csv(METRICAS / "comparison_original_vs_extended_behavioral.csv", index=False)
summary.to_csv(METRICAS / "comparison_original_vs_extended_behavioral_summary.csv", index=False)
importances.to_csv(METRICAS / "feature_importance_extended_rf.csv", header=["importancia"])
print("\nSalvo em comparison_original_vs_extended_behavioral.csv, ..._summary.csv e feature_importance_extended_rf.csv")
