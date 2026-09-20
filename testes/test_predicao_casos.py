import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predicao import prever_risco

base = {
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

print("--- Caso: variavel faltando ---")
incompleto = dict(base)
del incompleto["total_perdas"]
del incompleto["variedade_produtos"]
try:
    prever_risco(incompleto)
except ValueError as e:
    print(f"ValueError: {e}")

print("\n--- Caso: negativo invalido (total_apostado negativo) ---")
invalido = dict(base)
invalido["total_apostado"] = -500.0
try:
    prever_risco(invalido)
except ValueError as e:
    print(f"ValueError: {e}")

print("\n--- Caso: concentracao fora de [0,1] ---")
invalido2 = dict(base)
invalido2["concentracao_apostas_dias_ativos"] = 1.4
try:
    prever_risco(invalido2)
except ValueError as e:
    print(f"ValueError: {e}")

print("\n--- Caso: incoerencia entre intensidade e numero_total_apostas/dias_ativos ---")
incoerente = dict(base)
incoerente["intensidade_apostas_por_dia_ativo"] = 999.0  # deveria ser 3400/120 = 28.33
resultado = prever_risco(incoerente)
print(f"Classe: {resultado['classe_prevista']}")
print("Avisos:", resultado["avisos"])

print("\n--- Caso: valor acima do maximo observado no treino ---")
extremo = dict(base)
extremo["total_apostado"] = 999_999_999.0  # muito acima do maximo de treino (~12.1 milhoes)
resultado = prever_risco(extremo)
print(f"Classe: {resultado['classe_prevista']}")
print("Avisos:", resultado["avisos"])
