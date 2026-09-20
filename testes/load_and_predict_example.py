"""
Carrega o modelo salvo em modelo/ e faz uma predicao de exemplo, apenas para
verificar que a serializacao (joblib + metadata.json) funciona corretamente.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "modelo"

pipeline = joblib.load(MODEL_DIR / "xgboost_tunado_comportamental.joblib")
with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features_entrada_ordem"]
print(f"Modelo carregado. Features esperadas ({len(features)}): {features}")

# usuario de exemplo (valores fabricados, apenas para testar o pipeline)
exemplo = pd.DataFrame([{
    "total_dias_ativos": 120,
    "total_apostado": 15000.0,
    "total_perdas": 2200.0,
    "numero_total_apostas": 3400,
    "variedade_produtos": 4,
    "media_diaria_apostada": 125.0,
    "desvio_padrao_apostado": 300.0,
    "taxa_perda_sobre_apostado": 0.15,
    "intensidade_apostas_por_dia_ativo": 28.3,
    "tendencia_crescimento_apostado": 0.5,
    "concentracao_apostas_dias_ativos": 0.6,
    "media_apostas_por_produto": 850.0,
}])[features]  # reordena conforme metadata, por seguranca

classe_prevista = pipeline.predict(exemplo)[0]
probabilidades = pipeline.predict_proba(exemplo)[0]

classes_modelo = list(pipeline.named_steps["clf"].classes_)
label_para_nome = {v: k for k, v in metadata["classes"]["mapeamento_label_encoder"].items()}
nome_previsto = label_para_nome[int(classe_prevista)]

print(f"\nClasse prevista: {nome_previsto}")
print("Probabilidades por classe:")
for classe_idx, proba in zip(classes_modelo, probabilidades):
    print(f"  {label_para_nome[int(classe_idx)]}: {proba:.4f}")
