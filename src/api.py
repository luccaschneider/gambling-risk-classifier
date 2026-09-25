"""
API REST do modelo oficial de classificacao de risco (XGBoost tunado,
12 variaveis comportamentais).

Instalacao:
    pip install -r requirements.txt

Execucao (a partir da raiz do projeto):
    uvicorn src.api:app --reload --port 8000

Painel web do operador (HTML em static/index.html, consome POST /classificar):
    http://127.0.0.1:8000/

Documentacao interativa (Swagger UI):
    http://127.0.0.1:8000/docs

Exemplo de chamada:
    curl -X POST http://127.0.0.1:8000/classificar \
      -H "Content-Type: application/json" \
      -d '{
            "total_dias_ativos": 120,
            "total_apostado": 15000.0,
            "total_perdas": 2200.0,
            "numero_total_apostas": 3400,
            "variedade_produtos": 4,
            "media_diaria_apostada": 125.0,
            "desvio_padrao_apostado": 300.0,
            "taxa_perda_sobre_apostado": 0.15,
            "intensidade_apostas_por_dia_ativo": 28.33,
            "tendencia_crescimento_apostado": 0.5,
            "concentracao_apostas_dias_ativos": 0.6,
            "media_apostas_por_produto": 850.0
          }'
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# garante que predicao.py seja encontrado mesmo quando o uvicorn e iniciado
# de fora desta pasta (ex.: "uvicorn src.api:app" a partir da raiz)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import predicao
import registro

app = FastAPI(
    title="API de Classificacao de Risco - bwin",
    description="Expõe o modelo oficial (XGBoost tunado, 12 variaveis comportamentais) para classificacao de risco de jogo.",
    version="1.0.0",
)

# painel web (HTML estatico que consome POST /classificar via fetch)
_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_PAINEL = _STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

registro.iniciar()


# ============================================================================
# Modelos Pydantic (validacao de tipos de entrada/saida)
# ============================================================================
class EntradaClassificacao(BaseModel):
    total_dias_ativos: float = Field(..., description="Numero de dias distintos com aposta registrada")
    total_apostado: float = Field(..., description="Soma do valor apostado (Turnover) no periodo")
    total_perdas: float = Field(..., description="Soma do Hold no periodo (pode ser negativo)")
    numero_total_apostas: float = Field(..., description="Soma do numero de apostas (NumberofBets) no periodo")
    variedade_produtos: float = Field(..., description="Quantidade de tipos de produto distintos jogados")
    media_diaria_apostada: float = Field(..., description="Media do valor apostado por dia ativo")
    desvio_padrao_apostado: float = Field(..., description="Desvio padrao do valor apostado diario")
    taxa_perda_sobre_apostado: float = Field(..., description="total_perdas / total_apostado")
    intensidade_apostas_por_dia_ativo: float = Field(..., description="numero_total_apostas / total_dias_ativos")
    tendencia_crescimento_apostado: float = Field(..., description="Inclinacao da tendencia do valor apostado ao longo do tempo")
    concentracao_apostas_dias_ativos: float = Field(..., description="Indice de Gini da distribuicao diaria do valor apostado (0 a 1)")
    media_apostas_por_produto: float = Field(..., description="numero_total_apostas / variedade_produtos")

    usuario_externo: Optional[str] = Field(
        None,
        max_length=120,
        description=(
            "Identificador do usuario no sistema de origem. Opcional, gravado no "
            "registro auditavel para rastrear a decisao ate o cadastro da operadora."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_dias_ativos": 120,
                "total_apostado": 15000.0,
                "total_perdas": 2200.0,
                "numero_total_apostas": 3400,
                "variedade_produtos": 4,
                "media_diaria_apostada": 125.0,
                "desvio_padrao_apostado": 300.0,
                "taxa_perda_sobre_apostado": 0.15,
                "intensidade_apostas_por_dia_ativo": 28.33,
                "tendencia_crescimento_apostado": 0.5,
                "concentracao_apostas_dias_ativos": 0.6,
                "media_apostas_por_produto": 850.0,
            }
        }


class AcaoIntervencao(BaseModel):
    tipo: str = Field(..., description="Codigo da acao a executar, estavel entre versoes")
    alvo: str = Field(..., description="Sobre o que a acao incide: usuario, conta ou equipe_interna")
    bloqueia_conta: bool = Field(..., description="Indica se a acao impede o uso da conta")
    parametros: Dict[str, Any] = Field(..., description="Valores de referencia, definidos pela operadora")
    conteudo: Optional[str] = None
    carater: Optional[str] = None


class Intervencao(BaseModel):
    codigo: str = Field(..., description="Codigo da intervencao para o nivel de risco")
    nivel: str
    acoes: List[AcaoIntervencao] = Field(..., description="Acoes a executar, na ordem")
    descricao: str = Field(..., description="Texto para leitura humana, nao para processamento")


class ResultadoClassificacao(BaseModel):
    classe_prevista: str
    probabilidades: Dict[str, float]
    indice_risco: float = Field(
        ..., description="Posicao de 0 a 1 na escala dos tres niveis, derivada das probabilidades"
    )
    intervencao: Intervencao
    avisos: List[str]
    registro_id: Optional[int] = Field(
        None, description="Identificador da classificacao no registro auditavel"
    )


class RegistroClassificacao(BaseModel):
    id: int
    criado_em: str = Field(..., description="Data e hora em UTC, formato ISO 8601")
    usuario_externo: Optional[str] = None
    entradas: Dict[str, float]
    classe_prevista: str
    probabilidades: Dict[str, float]
    indice_risco: float
    intervencao_codigo: str
    avisos: List[str]
    versao_modelo: str


class RegistrosResposta(BaseModel):
    total_guardado: int = Field(..., description="Quantidade de classificacoes no registro")
    limite_tabela: int = Field(..., description="Maximo de entradas mantidas; as mais antigas sao apagadas")
    registros: List[RegistroClassificacao]


class SaudeResposta(BaseModel):
    status: str
    modelo_carregado: bool
    descricao: Optional[str] = None
    n_variaveis_esperadas: Optional[int] = None


class VariavelInfo(BaseModel):
    nome: str
    tipo: str
    minimo_observado_treino: Optional[float] = None
    maximo_observado_treino: Optional[float] = None


class VariaveisResposta(BaseModel):
    variaveis: List[VariavelInfo]


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/", include_in_schema=False)
def painel():
    """Serve o painel web na raiz. A documentacao continua em /docs."""
    return FileResponse(_PAINEL)


@app.post("/classificar", response_model=ResultadoClassificacao)
def classificar(entrada: EntradaClassificacao):
    """Recebe as 12 variaveis comportamentais e devolve a classificacao de risco.

    A intervencao vem estruturada em codigo e lista de acoes, para ser executada
    sem interpretar texto. Cada chamada e gravada no registro auditavel.
    """
    dados = entrada.model_dump()
    usuario_externo = dados.pop("usuario_externo", None)

    try:
        resultado = predicao.prever_risco(dados)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _, metadata = predicao._carregar()
    resultado["registro_id"] = registro.gravar(
        entradas=dados,
        resultado=resultado,
        indice_risco=resultado["indice_risco"],
        versao_modelo=metadata.get("versao_modelo", "desconhecida"),
        usuario_externo=usuario_externo,
    )
    return resultado


@app.get("/registros", response_model=RegistrosResposta)
def registros(
    limite: int = Query(
        20, ge=1, le=registro.ENTRADAS_MAXIMAS,
        description="Quantidade de classificacoes a devolver, da mais recente para a mais antiga",
    )
):
    """Lista as classificacoes ja feitas, para auditoria das decisoes.

    Cada entrada traz as variaveis recebidas, o resultado, a intervencao aplicada
    e a versao do modelo. Nesta demonstracao o registro fica em arquivo temporario
    e e apagado quando o servico reinicia ou hiberna; alem disso guarda apenas as
    ultimas entradas, entao nao serve como historico completo.
    """
    return RegistrosResposta(
        total_guardado=registro.total(),
        limite_tabela=registro.ENTRADAS_MAXIMAS,
        registros=registro.listar(limite),
    )


@app.get("/saude", response_model=SaudeResposta)
def saude():
    """Confirma que o modelo e os metadados estao carregados e prontos para uso."""
    try:
        _, metadata = predicao._carregar()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Falha ao carregar o modelo: {e}")
    return SaudeResposta(
        status="ok",
        modelo_carregado=True,
        descricao=metadata.get("descricao"),
        n_variaveis_esperadas=len(metadata["features_entrada_ordem"]),
    )


@app.get("/variaveis", response_model=VariaveisResposta)
def variaveis():
    """Lista, na ordem esperada pelo modelo, as 12 variaveis de entrada com tipo e faixa observada no treino."""
    try:
        _, metadata = predicao._carregar()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Falha ao carregar o modelo: {e}")

    features = metadata["features_entrada_ordem"]
    stats = metadata.get("estatisticas_treino", {})

    lista = [
        VariavelInfo(
            nome=f,
            tipo="float",
            minimo_observado_treino=stats.get(f, {}).get("min"),
            maximo_observado_treino=stats.get(f, {}).get("max"),
        )
        for f in features
    ]
    return VariaveisResposta(variaveis=lista)
