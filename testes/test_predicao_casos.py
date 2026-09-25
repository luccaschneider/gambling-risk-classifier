"""
Testa prever_risco: validacoes de entrada, avisos nao bloqueantes e a estrutura
da intervencao. Cada verificacao usa assert, entao o script falha sozinho.

Execucao:
    python testes/test_predicao_casos.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intervencoes import INTERVENCOES
from predicao import prever_risco

BASE = {
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

falhas = []


def checar(descricao, condicao, detalhe=""):
    if condicao:
        print(f"  ok    {descricao}")
    else:
        print(f"  FALHA {descricao}  {detalhe}")
        falhas.append(descricao)


print("== validacoes que devem bloquear ==")

try:
    incompleto = {k: v for k, v in BASE.items() if k not in {"total_perdas", "variedade_produtos"}}
    prever_risco(incompleto)
    checar("variaveis faltando levanta ValueError", False)
except ValueError as e:
    msg = str(e)
    checar("variaveis faltando levanta ValueError", True)
    checar("  erro lista as duas que faltam",
           "total_perdas" in msg and "variedade_produtos" in msg, msg[:80])

try:
    prever_risco({**BASE, "total_apostado": -500.0})
    checar("negativo invalido levanta ValueError", False)
except ValueError as e:
    checar("negativo invalido levanta ValueError", True)
    checar("  erro nomeia a variavel", "total_apostado" in str(e))

try:
    prever_risco({**BASE, "concentracao_apostas_dias_ativos": 1.4})
    checar("concentracao fora de [0,1] levanta ValueError", False)
except ValueError as e:
    checar("concentracao fora de [0,1] levanta ValueError", True)

# tendencia negativa e legitima: a mediana do treino e negativa
r = prever_risco({**BASE, "tendencia_crescimento_apostado": -0.4839530826079938})
checar("tendencia negativa e aceita", r["classe_prevista"] in INTERVENCOES)

print()
print("== avisos nao bloqueantes ==")

r = prever_risco({**BASE, "intensidade_apostas_por_dia_ativo": 999.0})
checar("incoerencia de intensidade gera aviso", len(r["avisos"]) >= 1)
checar("  aviso cita a variavel",
       any("intensidade_apostas_por_dia_ativo" in a for a in r["avisos"]))
checar("  a predicao continua valendo", r["classe_prevista"] in INTERVENCOES)

r = prever_risco({**BASE, "total_apostado": 999_999_999.0})
checar("valor acima do maximo de treino gera aviso",
       any("acima do maximo" in a for a in r["avisos"]))

print()
print("== estrutura da intervencao ==")

for classe, esperado in [("Baixo risco", "MONITORAMENTO_INFORMATIVO"),
                         ("Medio risco", "LIMITE_TEMPORARIO"),
                         ("Alto risco", "SUSPENSAO_TEMPORARIA")]:
    intervencao = INTERVENCOES[classe]
    checar(f"{classe}: codigo {esperado}", intervencao["codigo"] == esperado, intervencao["codigo"])
    checar(f"{classe}: tem acoes", len(intervencao["acoes"]) >= 1)
    checar(f"{classe}: descricao acentuada",
           any(c in intervencao["descricao"] for c in "áéíóúâêôãõç"))
    for acao in intervencao["acoes"]:
        checar(f"{classe}: acao {acao['tipo']} tem campos obrigatorios",
               {"tipo", "alvo", "bloqueia_conta", "parametros"} <= set(acao))

bloqueiam = {c: any(a["bloqueia_conta"] for a in i["acoes"]) for c, i in INTERVENCOES.items()}
checar("so o alto risco bloqueia a conta",
       bloqueiam == {"Baixo risco": False, "Medio risco": False, "Alto risco": True}, str(bloqueiam))

resultado = prever_risco(BASE)
checar("resultado traz intervencao estruturada", isinstance(resultado["intervencao"], dict))
checar("  intervencao corresponde a classe prevista",
       resultado["intervencao"]["codigo"] == INTERVENCOES[resultado["classe_prevista"]]["codigo"])

print()
print("== indice de risco ==")

checar("indice esta entre 0 e 1", 0.0 <= resultado["indice_risco"] <= 1.0)
esperado = (resultado["probabilidades"]["Medio risco"] * 0.5
            + resultado["probabilidades"]["Alto risco"])
checar("indice confere com a formula declarada",
       abs(resultado["indice_risco"] - esperado) < 1e-9)

baixo = prever_risco({**BASE, "total_apostado": 50.0, "numero_total_apostas": 20,
                      "total_dias_ativos": 10, "intensidade_apostas_por_dia_ativo": 2.0,
                      "media_apostas_por_produto": 5.0, "variedade_produtos": 4})
checar("perfil de baixa intensidade tem indice menor",
       baixo["indice_risco"] < resultado["indice_risco"],
       f"{baixo['indice_risco']:.3f} vs {resultado['indice_risco']:.3f}")

print()
if falhas:
    print(f"FALHARAM {len(falhas)} verificacoes:")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("todas as verificacoes passaram")
