import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

df = pd.read_csv(DADOS / "bwin_dataset_agregado_classificado.csv")

# --- features e target -----------------------------------------------------
# Excluidas por vazamento de rotulo (foram usadas para CONSTRUIR a classificacao):
#   RG_case, RGsumevents, RGFirst_Date, RGLast_date, Event_type_first,
#   Interventiontype_first  -> determinam diretamente "Alto risco"
#   escore_intensidade      -> determina diretamente "Medio" vs "Baixo" risco
#   UserID                  -> identificador, sem valor preditivo
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

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, random_state=42, eval_metric="mlogloss",
        tree_method="hist",
    ),
}

y_test_bin = label_binarize(y_test, classes=le.transform(class_order))

rows = []
summary_rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    # reordena colunas de proba conforme class_order
    class_idx = [list(pipe.classes_).index(c) for c in le.transform(class_order)]
    y_proba_ord = y_proba[:, class_idx]

    f1_per_class = f1_score(y_test, y_pred, labels=le.transform(class_order), average=None)
    rec_per_class = recall_score(y_test, y_pred, labels=le.transform(class_order), average=None)
    auc_per_class = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")

    for i, cls in enumerate(class_order):
        rows.append({
            "Modelo": name, "Classe": cls,
            "AUC": auc_per_class[i], "F1": f1_per_class[i], "Recall": rec_per_class[i],
        })

    auc_macro = roc_auc_score(y_test_bin, y_proba_ord, average="macro", multi_class="ovr")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    acc = (y_pred == y_test).mean()
    summary_rows.append({
        "Modelo": name, "Acuracia": acc, "AUC_macro": auc_macro,
        "F1_macro": f1_macro, "Recall_macro": rec_macro,
    })

detail = pd.DataFrame(rows)
summary = pd.DataFrame(summary_rows).sort_values("AUC_macro", ascending=False)

pd.set_option("display.width", 140)
print("=== Metricas por modelo e classe de risco ===")
print(detail.round(4).to_string(index=False))

print("\n=== Resumo comparativo (medias macro no conjunto de teste) ===")
print(summary.round(4).to_string(index=False))

detail.to_csv(METRICAS / "model_comparison_by_class.csv", index=False)
summary.to_csv(METRICAS / "model_comparison_summary.csv", index=False)
print("\nSalvo em model_comparison_by_class.csv e model_comparison_summary.csv")
