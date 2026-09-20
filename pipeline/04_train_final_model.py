"""
Treina o modelo OFICIAL final (XGBoost tunado, 12 variaveis comportamentais,
sem balanceamento) e salva o pipeline completo + metadados na pasta modelo/,
para reuso sem precisar retreinar.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
MODELO = RAIZ / "modelo"

MODELO.mkdir(exist_ok=True)

df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")

BEHAVIORAL_FEATURES = [
    "total_dias_ativos", "total_apostado", "total_perdas", "numero_total_apostas",
    "variedade_produtos", "media_diaria_apostada", "desvio_padrao_apostado",
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
]
CLASS_ORDER = ["Baixo risco", "Medio risco", "Alto risco"]

X = df[BEHAVIORAL_FEATURES]
y = df["classificacao_risco"]

le = LabelEncoder()
le.fit(CLASS_ORDER)
y_enc = le.transform(y)
class_idx_order = le.transform(CLASS_ORDER)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

XGB_PARAMS = dict(
    n_estimators=279, max_depth=3, learning_rate=0.02069721473281451,
    subsample=0.7644148053272926, colsample_bytree=0.7203513239267079,
    min_child_weight=2, gamma=0.11393619775098705,
    random_state=42, eval_metric="mlogloss", tree_method="hist",
)

pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", XGBClassifier(**XGB_PARAMS)),
])
pipeline.fit(X_train, y_train)

# --- metricas no teste (mesmas do modelo oficial ja reportado) --------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
col_order = [list(pipeline.named_steps["clf"].classes_).index(c) for c in class_idx_order]
y_proba_ord = y_proba[:, col_order]
y_test_bin = label_binarize(y_test, classes=class_idx_order)

f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
aucs = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")

metrics_per_class = {
    cls: {"AUC": float(aucs[i]), "F1": float(f1s[i]), "Recall": float(recs[i])}
    for i, cls in enumerate(CLASS_ORDER)
}
metrics_macro = {
    "AUC_macro": float(aucs.mean()),
    "F1_macro": float(f1s.mean()),
    "Recall_macro": float(recs.mean()),
}

# --- salva o pipeline completo (pre-processamento + modelo) em um unico arquivo ---
model_path = MODELO / "xgboost_tunado_comportamental.joblib"
joblib.dump(pipeline, model_path)
print(f"Pipeline salvo em: {model_path}")

# --- metadados ----------------------------------------------------------------
metadata = {
    "descricao": "XGBoost tunado, 12 variaveis comportamentais, sem balanceamento (modelo oficial)",
    "features_entrada_ordem": BEHAVIORAL_FEATURES,
    "classes": {
        "ordem_risco": CLASS_ORDER,
        "mapeamento_label_encoder": {cls: int(le.transform([cls])[0]) for cls in CLASS_ORDER},
    },
    "split": {"test_size": 0.20, "random_state": 42, "stratify": True},
    "hiperparametros_xgboost": XGB_PARAMS,
    "metricas_teste": {
        "macro": metrics_macro,
        "por_classe": metrics_per_class,
    },
}

metadata_path = MODELO / "metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Metadados salvos em: {metadata_path}")

print("\nResumo das metricas de teste salvas:")
print(json.dumps(metadata["metricas_teste"], indent=2, ensure_ascii=False))
