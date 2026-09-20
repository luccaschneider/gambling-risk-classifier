import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import app

client = TestClient(app)

print("--- GET /saude ---")
r = client.get("/saude")
print(r.status_code, r.json())

print("\n--- GET /variaveis ---")
r = client.get("/variaveis")
print(r.status_code)
for v in r.json()["variaveis"]:
    print(" ", v)

print("\n--- POST /classificar (caso valido) ---")
payload_valido = {
    "total_dias_ativos": 120,
    "total_apostado": 15000.0,
    "total_perdas": 2200.0,
    "numero_total_apostas": 3400,
    "variedade_produtos": 4,
    "media_diaria_apostada": 125.0,
    "desvio_padrao_apostado": 300.0,
    "taxa_perda_sobre_apostado": 0.15,
    "intensidade_apostas_por_dia_ativo": 3400 / 120,
    "tendencia_crescimento_apostado": 0.5,
    "concentracao_apostas_dias_ativos": 0.6,
    "media_apostas_por_produto": 3400 / 4,
}
r = client.post("/classificar", json=payload_valido)
print(r.status_code, r.json())

print("\n--- POST /classificar (variavel faltando -> deve dar 422 do Pydantic) ---")
payload_faltando = dict(payload_valido)
del payload_faltando["total_perdas"]
r = client.post("/classificar", json=payload_faltando)
print(r.status_code, r.json())

print("\n--- POST /classificar (tipo invalido -> deve dar 422 do Pydantic) ---")
payload_tipo_invalido = dict(payload_valido)
payload_tipo_invalido["total_apostado"] = "nao-e-numero"
r = client.post("/classificar", json=payload_tipo_invalido)
print(r.status_code, r.json())

print("\n--- POST /classificar (negativo invalido -> deve dar 422 da funcao prever_risco) ---")
payload_negativo = dict(payload_valido)
payload_negativo["total_apostado"] = -500.0
r = client.post("/classificar", json=payload_negativo)
print(r.status_code, r.json())

print("\n--- POST /classificar (concentracao fora de [0,1] -> deve dar 422 da funcao prever_risco) ---")
payload_concentracao = dict(payload_valido)
payload_concentracao["concentracao_apostas_dias_ativos"] = 1.5
r = client.post("/classificar", json=payload_concentracao)
print(r.status_code, r.json())

print("\n--- POST /classificar (tendencia NEGATIVA -> deve ser ACEITA, status 200) ---")
payload_tendencia_negativa = dict(payload_valido)
payload_tendencia_negativa["tendencia_crescimento_apostado"] = -0.4839530826079938
r = client.post("/classificar", json=payload_tendencia_negativa)
print(r.status_code, r.json())
assert r.status_code == 200, "tendencia_crescimento_apostado negativa deveria ser aceita"

print("\n--- POST /classificar (total_dias_ativos NEGATIVO -> deve continuar sendo 422) ---")
payload_dias_negativo = dict(payload_valido)
payload_dias_negativo["total_dias_ativos"] = -10
r = client.post("/classificar", json=payload_dias_negativo)
print(r.status_code, r.json())
assert r.status_code == 422, "total_dias_ativos negativo deveria ser rejeitado"
