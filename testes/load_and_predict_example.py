"""
Exemplo de uso do modelo serializado, carregando joblib e metadata.json
diretamente — sem passar pela funcao prever_risco.

Serve como referencia de integracao e, ao mesmo tempo, como verificacao: se o
carregamento ou a predicao falharem, o script termina com codigo de erro.

Execucao:
    python testes/load_and_predict_example.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "modelo"

falhas = []


def checar(descricao, condicao, detalhe=""):
    if condicao:
        print(f"  ok    {descricao}")
    else:
        print(f"  FALHA {descricao}  {detalhe}")
        falhas.append(descricao)


# ---------------------------------------------------------------- carregamento
print("== carregamento ==")
checar("arquivo do modelo existe", (MODEL_DIR / "xgboost_tunado_comportamental.joblib").exists())
checar("metadata.json existe", (MODEL_DIR / "metadata.json").exists())

pipeline = joblib.load(MODEL_DIR / "xgboost_tunado_comportamental.joblib")
with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features_entrada_ordem"]
checar("pipeline carregado", pipeline is not None)
checar("metadata declara 12 variaveis", len(features) == 12, len(features))
checar("pipeline tem as etapas esperadas",
       {"impute", "scale", "clf"} <= set(pipeline.named_steps), list(pipeline.named_steps))
checar("metadata traz a versao do modelo", bool(metadata.get("versao_modelo")), metadata.get("versao_modelo"))

print(f"\n  variaveis esperadas ({len(features)}): {', '.join(features)}")

# ------------------------------------------------------------------- predicao
print()
print("== predicao de exemplo ==")

# usuario de exemplo (valores fabricados, apenas para exercitar o pipeline)
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
por_classe = {label_para_nome[int(c)]: float(p) for c, p in zip(classes_modelo, probabilidades)}

print(f"  classe prevista: {nome_previsto}")
for classe, proba in por_classe.items():
    print(f"    {classe}: {proba:.4f}")

checar("classe prevista e uma das tres", nome_previsto in metadata["classes"]["ordem_risco"], nome_previsto)
checar("ha uma probabilidade por classe", len(por_classe) == 3, len(por_classe))
checar("probabilidades somam 1", abs(sum(por_classe.values()) - 1.0) < 1e-6, sum(por_classe.values()))
checar("a classe prevista e a de maior probabilidade",
       max(por_classe, key=por_classe.get) == nome_previsto)
checar("perfil de alta intensidade cai em alto risco", nome_previsto == "Alto risco", nome_previsto)

# ----------------------------------------------------- ordem das variaveis
print()
print("== a ordem das variaveis importa ==")
# o proprio sklearn compara os nomes das colunas com os do treino e recusa a
# entrada se a ordem diferir — por isso o exemplo acima faz [features]
invertido = exemplo[list(reversed(features))]
try:
    pipeline.predict_proba(invertido)
    checar("colunas fora de ordem sao recusadas", False, "a predicao passou sem erro")
except ValueError as e:
    checar("colunas fora de ordem sao recusadas pelo sklearn", True)
    print(f"  erro esperado: {str(e).splitlines()[0]}")

print()
if falhas:
    print(f"FALHARAM {len(falhas)} verificacoes:")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("todas as verificacoes passaram")
