"""
Valida a serializacao do modelo salvo em modelo/:
1) recarrega o pipeline e reavalia no MESMO split de teste (80/20,
   random_state=42) usado no treino original, comparando com metadata.json;
2) testa 3 casos limite: zeros, valores extremos, e valores faltantes.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
MODEL_DIR = RAIZ / "modelo"

pipeline = joblib.load(MODEL_DIR / "xgboost_tunado_comportamental.joblib")
with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features_entrada_ordem"]
class_order = metadata["classes"]["ordem_risco"]

df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")
X = df[features]
y = df["classificacao_risco"]

le = LabelEncoder()
le.fit(class_order)
y_enc = le.transform(y)
class_idx_order = le.transform(class_order)

# mesmo split usado no treino original (test_size, random_state e stratify identicos)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=metadata["split"]["test_size"],
    random_state=metadata["split"]["random_state"], stratify=y_enc
)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
col_order = [list(pipeline.named_steps["clf"].classes_).index(c) for c in class_idx_order]
y_proba_ord = y_proba[:, col_order]
y_test_bin = label_binarize(y_test, classes=class_idx_order)

f1s = f1_score(y_test, y_pred, labels=class_idx_order, average=None)
recs = recall_score(y_test, y_pred, labels=class_idx_order, average=None)
aucs = roc_auc_score(y_test_bin, y_proba_ord, average=None, multi_class="ovr")

recomputed = {
    "macro": {
        "AUC_macro": float(aucs.mean()),
        "F1_macro": float(f1s.mean()),
        "Recall_macro": float(recs.mean()),
    },
    "por_classe": {
        cls: {"AUC": float(aucs[i]), "F1": float(f1s[i]), "Recall": float(recs[i])}
        for i, cls in enumerate(class_order)
    },
}

saved = metadata["metricas_teste"]

print("=== 1) Comparacao: metricas recalculadas (modelo carregado) x metadata.json ===\n")

print("-- Macro --")
tudo_igual = True
for k in recomputed["macro"]:
    v_saved = saved["macro"][k]
    v_new = recomputed["macro"][k]
    igual = np.isclose(v_saved, v_new, atol=1e-12)
    tudo_igual &= igual
    print(f"  {k}: salvo={v_saved:.10f}  recalculado={v_new:.10f}  identico={igual}")

print("\n-- Por classe --")
for cls in class_order:
    print(f"\n{cls}:")
    for metric in ["AUC", "F1", "Recall"]:
        v_saved = saved["por_classe"][cls][metric]
        v_new = recomputed["por_classe"][cls][metric]
        igual = np.isclose(v_saved, v_new, atol=1e-12)
        tudo_igual &= igual
        print(f"  {metric}: salvo={v_saved:.10f}  recalculado={v_new:.10f}  identico={igual}")

print(f"\n=> RESULTADO: {'TODAS as metricas batem exatamente' if tudo_igual else 'DIVERGENCIA ENCONTRADA'}")

# ============================================================================
# 2) CASOS LIMITE
# ============================================================================
print("\n\n=== 2) Casos limite ===\n")


def rodar_caso(nome, linha_dict):
    print(f"--- {nome} ---")
    entrada = pd.DataFrame([linha_dict])[features]
    print(entrada.to_string(index=False))
    try:
        pred = pipeline.predict(entrada)[0]
        proba = pipeline.predict_proba(entrada)[0]
        classes_modelo = list(pipeline.named_steps["clf"].classes_)
        label_para_nome = {v: k for k, v in metadata["classes"]["mapeamento_label_encoder"].items()}
        nome_previsto = label_para_nome[int(pred)]
        print(f"Predicao: {nome_previsto}")
        for c_idx, p in zip(classes_modelo, proba):
            print(f"  {label_para_nome[int(c_idx)]}: {p:.4f}")
    except Exception as e:
        print(f"ERRO: {type(e).__name__}: {e}")
    print()


# caso 1: tudo zero
caso_zeros = {f: 0.0 for f in features}
rodar_caso("Caso 1 - todos os valores em zero", caso_zeros)

# caso 2: valores extremamente altos (muito acima do maximo observado no dataset)
maximos = df[features].max()
caso_extremo = {f: float(maximos[f]) * 100 for f in features}
rodar_caso("Caso 2 - valores extremamente altos (100x o maximo observado)", caso_extremo)

# caso 3: valores faltantes em 3 variaveis (usa um caso "normal" com 3 NaN)
caso_faltante = {
    "total_dias_ativos": 90,
    "total_apostado": 5000.0,
    "total_perdas": np.nan,
    "numero_total_apostas": 800,
    "variedade_produtos": np.nan,
    "media_diaria_apostada": 55.0,
    "desvio_padrao_apostado": 120.0,
    "taxa_perda_sobre_apostado": np.nan,
    "intensidade_apostas_por_dia_ativo": 8.9,
    "tendencia_crescimento_apostado": 0.1,
    "concentracao_apostas_dias_ativos": 0.5,
    "media_apostas_por_produto": 200.0,
}
rodar_caso("Caso 3 - valores faltantes em 3 variaveis (total_perdas, variedade_produtos, taxa_perda_sobre_apostado)", caso_faltante)
