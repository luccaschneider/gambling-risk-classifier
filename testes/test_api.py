"""
Testa a API: classificacao, validacoes, intervencao estruturada, gravacao no
registro auditavel e o endpoint de consulta. Cada verificacao usa assert.

Usa um banco proprio num arquivo temporario, para nao misturar com o registro
de uma instancia em execucao.

Execucao:
    python testes/test_api.py
"""
import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
os.environ["REGISTRO_BANCO"] = str(Path(tempfile.mkdtemp()) / "teste_registro.sqlite3")
sys.path.insert(0, str(RAIZ / "src"))

from fastapi.testclient import TestClient

from api import app
import registro

cliente = TestClient(app)

PAYLOAD = {
    "total_dias_ativos": 157,
    "total_apostado": 43172.61,
    "total_perdas": 1550.66,
    "numero_total_apostas": 7305,
    "variedade_produtos": 4,
    "media_diaria_apostada": 274.984777,
    "desvio_padrao_apostado": 483.319542,
    "taxa_perda_sobre_apostado": 0.035918,
    "intensidade_apostas_por_dia_ativo": 46.528662,
    "tendencia_crescimento_apostado": -0.483953,
    "concentracao_apostas_dias_ativos": 0.69478,
    "media_apostas_por_produto": 1826.25,
}

falhas = []


def checar(descricao, condicao, detalhe=""):
    if condicao:
        print(f"  ok    {descricao}")
    else:
        print(f"  FALHA {descricao}  {detalhe}")
        falhas.append(descricao)


print("== endpoints basicos ==")
r = cliente.get("/saude")
checar("GET /saude responde 200", r.status_code == 200, r.status_code)
checar("  modelo carregado", r.json().get("modelo_carregado") is True)

r = cliente.get("/variaveis")
checar("GET /variaveis responde 200", r.status_code == 200)
checar("  lista as 12 variaveis", len(r.json()["variaveis"]) == 12)

r = cliente.get("/")
checar("GET / serve o painel", r.status_code == 200 and "text/html" in r.headers["content-type"])

r = cliente.get("/docs")
checar("GET /docs serve o Swagger", r.status_code == 200)

print()
print("== classificacao e intervencao estruturada ==")
r = cliente.post("/classificar", json=PAYLOAD)
checar("POST /classificar responde 200", r.status_code == 200, r.text[:120])
corpo = r.json()
checar("  classe prevista e Alto risco", corpo["classe_prevista"] == "Alto risco", corpo["classe_prevista"])
checar("  traz indice_risco entre 0 e 1", 0.0 <= corpo["indice_risco"] <= 1.0)
checar("  traz registro_id", isinstance(corpo.get("registro_id"), int))

intervencao = corpo["intervencao"]
checar("  intervencao e objeto, nao texto", isinstance(intervencao, dict))
checar("  tem codigo, acoes e descricao",
       {"codigo", "acoes", "descricao", "nivel"} <= set(intervencao))
checar("  codigo do alto risco", intervencao["codigo"] == "SUSPENSAO_TEMPORARIA", intervencao["codigo"])
checar("  acoes do alto risco",
       [a["tipo"] for a in intervencao["acoes"]] == ["SUSPENDER_CONTA", "ENCAMINHAR_SUPORTE"])
checar("  acao de suspensao bloqueia a conta", intervencao["acoes"][0]["bloqueia_conta"] is True)
checar("  parametros vem preenchidos", bool(intervencao["acoes"][0]["parametros"]))
checar("  descricao tem acentuacao", any(c in intervencao["descricao"] for c in "áéíóúâêôãõç"))

# o direito de recurso precisa sobreviver a serializacao da API, nao so existir no modulo
suspensao = next(a for a in intervencao["acoes"] if a["tipo"] == "SUSPENDER_CONTA")
encaminhamento = next(a for a in intervencao["acoes"] if a["tipo"] == "ENCAMINHAR_SUPORTE")
checar("  suspensao permite contestacao",
       suspensao["parametros"].get("contestacao_permitida") is True)
checar("  suspensao traz prazo de reanalise",
       isinstance(suspensao["parametros"].get("prazo_resposta_reanalise_dias"), int))
checar("  encaminhamento comunica o direito de recurso",
       encaminhamento["parametros"].get("comunicar_direito_recurso") is True)
checar("  descricao menciona contestacao e reanalise",
       "contestar" in intervencao["descricao"] and "reanálise" in intervencao["descricao"])

print()
print("== registro auditavel ==")
antes = registro.total()
r = cliente.post("/classificar", json={**PAYLOAD, "usuario_externo": "teste-abc"})
checar("classificacao com usuario_externo responde 200", r.status_code == 200)
checar("  o registro cresceu", registro.total() == antes + 1, f"{antes} -> {registro.total()}")

r = cliente.get("/registros?limite=5")
checar("GET /registros responde 200", r.status_code == 200)
dados = r.json()
checar("  traz total_guardado e limite_tabela",
       {"total_guardado", "limite_tabela", "registros"} <= set(dados))
checar("  limite da tabela e 500", dados["limite_tabela"] == 500, dados["limite_tabela"])

ultimo = dados["registros"][0]
checar("  o mais recente vem primeiro", ultimo["usuario_externo"] == "teste-abc", ultimo["usuario_externo"])
checar("  guarda as 12 entradas", len(ultimo["entradas"]) == 12, len(ultimo["entradas"]))
checar("  guarda a classe prevista", ultimo["classe_prevista"] == "Alto risco")
checar("  guarda o codigo da intervencao", ultimo["intervencao_codigo"] == "SUSPENSAO_TEMPORARIA")
checar("  guarda a versao do modelo", ultimo["versao_modelo"] == "1.0.0", ultimo["versao_modelo"])
checar("  guarda data em UTC (ISO 8601)", ultimo["criado_em"].endswith("+00:00"), ultimo["criado_em"])
checar("  guarda o indice", isinstance(ultimo["indice_risco"], float))

# usuario_externo e opcional
r = cliente.post("/classificar", json=PAYLOAD)
checar("usuario_externo e opcional", r.status_code == 200)
checar("  fica nulo quando ausente",
       cliente.get("/registros?limite=1").json()["registros"][0]["usuario_externo"] is None)

checar("limite fora da faixa e rejeitado", cliente.get("/registros?limite=0").status_code == 422)

print()
print("== validacoes de entrada ==")
sem_campo = {k: v for k, v in PAYLOAD.items() if k != "total_perdas"}
checar("variavel faltando -> 422 do Pydantic",
       cliente.post("/classificar", json=sem_campo).status_code == 422)
checar("tipo invalido -> 422 do Pydantic",
       cliente.post("/classificar", json={**PAYLOAD, "total_apostado": "texto"}).status_code == 422)

r = cliente.post("/classificar", json={**PAYLOAD, "total_apostado": -500.0})
checar("negativo invalido -> 422 da regra de negocio", r.status_code == 422)
checar("  mensagem e legivel", isinstance(r.json()["detail"], str) and "negativo" in r.json()["detail"])

checar("concentracao fora de [0,1] -> 422",
       cliente.post("/classificar", json={**PAYLOAD, "concentracao_apostas_dias_ativos": 1.5}).status_code == 422)
checar("tendencia negativa -> 200",
       cliente.post("/classificar", json={**PAYLOAD, "tendencia_crescimento_apostado": -0.48}).status_code == 200)
checar("total_dias_ativos negativo -> 422",
       cliente.post("/classificar", json={**PAYLOAD, "total_dias_ativos": -10}).status_code == 422)

print()
print("== entradas invalidas nao entram no registro ==")
antes = registro.total()
cliente.post("/classificar", json={**PAYLOAD, "total_apostado": -1})
checar("classificacao rejeitada nao e gravada", registro.total() == antes, f"{antes} -> {registro.total()}")

print()
print("== poda da tabela ==")
# grava direto no modulo para nao rodar o modelo centenas de vezes
resultado_falso = {
    "classe_prevista": "Baixo risco",
    "probabilidades": {"Baixo risco": 0.9, "Medio risco": 0.07, "Alto risco": 0.03},
    "intervencao": {"codigo": "MONITORAMENTO_INFORMATIVO"},
    "avisos": [],
}
for i in range(registro.ENTRADAS_MAXIMAS + 5):
    registro.gravar(PAYLOAD, resultado_falso, 0.065, "1.0.0", usuario_externo=f"poda-{i}")

checar("tabela para no limite de 500", registro.total() == registro.ENTRADAS_MAXIMAS, registro.total())
recentes = registro.listar(limite=registro.ENTRADAS_MAXIMAS)
checar("  o mais recente e o ultimo gravado",
       recentes[0]["usuario_externo"] == f"poda-{registro.ENTRADAS_MAXIMAS + 4}",
       recentes[0]["usuario_externo"])
checar("  as entradas antigas foram apagadas",
       all(r["usuario_externo"] != "teste-abc" for r in recentes))
checar("  ids sao decrescentes", [r["id"] for r in recentes] == sorted((r["id"] for r in recentes), reverse=True))

print()
if falhas:
    print(f"FALHARAM {len(falhas)} verificacoes:")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("todas as verificacoes passaram")
