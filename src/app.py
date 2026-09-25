"""
Interface de demonstracao (Streamlit) do modelo oficial de classificacao de
risco - consome predicao.prever_risco() diretamente, sem passar pela API.

Instalacao:
    pip install -r requirements.txt

Execucao (a partir da raiz do projeto):
    streamlit run src/app.py

A pagina abre automaticamente em http://localhost:8501
"""
import html
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# caminhos resolvidos a partir da raiz do projeto, para rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predicao import _carregar, prever_risco

st.set_page_config(
    page_title="Classificacao de risco comportamental em apostas online",
    layout="wide",
)

FEATURES = [
    "total_dias_ativos", "total_apostado", "total_perdas", "numero_total_apostas",
    "variedade_produtos", "media_diaria_apostada", "desvio_padrao_apostado",
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
]
CLASS_ORDER = ["Baixo risco", "Medio risco", "Alto risco"]
NOME_EXIBICAO = {"Baixo risco": "Baixo risco", "Medio risco": "Médio risco", "Alto risco": "Alto risco"}

# verde / ambar / vermelho - mesma convencao semantica dos graficos do TCC
CORES_RISCO = {"Baixo risco": "#1baf7a", "Medio risco": "#eda100", "Alto risco": "#e34948"}

SUPERFICIE = "#fcfcfb"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#6b6a66"
TRILHA = "#eeede8"

ROTULOS = {
    "total_dias_ativos": "Total de dias ativos",
    "total_apostado": "Total apostado (unidade monetária)",
    "total_perdas": "Total de perdas / Hold (unidade monetária)",
    "numero_total_apostas": "Número total de apostas",
    "variedade_produtos": "Variedade de produtos jogados",
    "media_diaria_apostada": "Média diária apostada",
    "desvio_padrao_apostado": "Desvio padrão do valor apostado",
    "taxa_perda_sobre_apostado": "Taxa de perda sobre o total apostado",
    "intensidade_apostas_por_dia_ativo": "Intensidade de apostas por dia ativo",
    "tendencia_crescimento_apostado": "Tendência de crescimento do valor apostado",
    "concentracao_apostas_dias_ativos": "Concentração das apostas nos dias ativos",
    "media_apostas_por_produto": "Média de apostas por produto",
}

AJUDA = {
    "total_dias_ativos": "Número de dias distintos em que o usuário fez pelo menos 1 aposta.",
    "total_apostado": "Soma de todo o valor apostado (Turnover) no período observado.",
    "total_perdas": "Soma do Hold (receita líquida da casa). Pode ser negativo se o usuário ganhou mais do que apostou.",
    "numero_total_apostas": "Soma da quantidade de apostas feitas em todo o período.",
    "variedade_produtos": "Quantidade de tipos de produto distintos que o usuário jogou (ex.: apostas esportivas, cassino, poker).",
    "media_diaria_apostada": "Valor médio apostado por dia, considerando apenas os dias em que houve atividade.",
    "desvio_padrao_apostado": "Quão irregular é o valor apostado dia a dia - valores altos indicam grande variação entre dias.",
    "taxa_perda_sobre_apostado": "total_perdas dividido por total_apostado - proporção do valor apostado que virou perda líquida.",
    "intensidade_apostas_por_dia_ativo": "numero_total_apostas dividido por total_dias_ativos - quantas apostas o usuário faz, em média, em cada dia ativo.",
    "tendencia_crescimento_apostado": "Inclinação da tendência do valor apostado diário ao longo do tempo. Positivo = apostas crescentes; negativo = decrescentes.",
    "concentracao_apostas_dias_ativos": "Índice de Gini (0 a 1) da distribuição do valor apostado entre os dias ativos. Perto de 1 = quase tudo concentrado em poucos dias.",
    "media_apostas_por_produto": "numero_total_apostas dividido por variedade_produtos - quantas apostas, em média, por tipo de produto jogado.",
}

# contagens inteiras / valores monetarios com 2 casas / indices e taxas com 4 casas
VARIAVEIS_INTEIRAS = {"total_dias_ativos", "numero_total_apostas", "variedade_produtos"}
VARIAVEIS_MONETARIAS = {
    "total_apostado", "total_perdas", "media_diaria_apostada", "desvio_padrao_apostado",
}
VARIAVEIS_QUATRO_CASAS = {
    "taxa_perda_sobre_apostado", "tendencia_crescimento_apostado",
    "concentracao_apostas_dias_ativos",
}


def formato_campo(feat):
    """Devolve (format, step) do number_input conforme a natureza da variavel."""
    if feat in VARIAVEIS_INTEIRAS:
        return "%d", 1
    if feat in VARIAVEIS_QUATRO_CASAS:
        return "%.4f", 0.0001
    return "%.2f", 0.01  # monetarias e demais razoes


def valor_tipado(feat, valor):
    """Converte para int nos campos inteiros (exigencia do format '%d')."""
    return int(round(float(valor))) if feat in VARIAVEIS_INTEIRAS else float(valor)


CSS = """
<style>
.block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1180px; }
html, body, [data-testid="stAppViewContainer"] {
    font-family: -apple-system, "Segoe UI", system-ui, Roboto, sans-serif;
}
h1 {
    font-size: 2.6rem !important; font-weight: 600 !important;
    letter-spacing: -0.02em; line-height: 1.15; color: #111111;
    margin-bottom: 0.35rem !important;
}
h2, h3 {
    font-size: 0.82rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: #52514e !important; margin-top: 0.4rem !important;
}
[data-testid="stCaptionContainer"] p { font-size: 0.85rem; color: #6b6a66; line-height: 1.5; }
.stButton button {
    border-radius: 3px; border: 1px solid #d4d3cd; font-weight: 500;
    font-size: 0.88rem; color: #2b2a28; padding: 0.4rem 0.9rem;
    background-color: #ffffff;
}
.stButton button:hover { border-color: #a9a8a2; color: #0b0b0b; }

/* botoes de perfil: borda na cor do nivel de risco, fundo branco e leve
   preenchimento da propria cor no hover. O texto usa um passo mais escuro da
   mesma matiz para alcancar contraste WCAG AA (>= 4.5:1) sobre o branco:
   #1baf7a -> #15875e (4.51:1), #eda100 -> #9e6b00 (4.61:1), #e34948 -> #df3130 (4.55:1) */
.st-key-perfil_baixo button, .st-key-perfil_baixo button:focus:not(:active) {
    border-color: #1baf7a; color: #15875e; background-color: #ffffff;
}
.st-key-perfil_baixo button:hover {
    border-color: #1baf7a; color: #15875e; background-color: rgba(27, 175, 122, 0.10);
}
.st-key-perfil_medio button, .st-key-perfil_medio button:focus:not(:active) {
    border-color: #eda100; color: #9e6b00; background-color: #ffffff;
}
.st-key-perfil_medio button:hover {
    border-color: #eda100; color: #9e6b00; background-color: rgba(237, 161, 0, 0.10);
}
.st-key-perfil_alto button, .st-key-perfil_alto button:focus:not(:active) {
    border-color: #e34948; color: #df3130; background-color: #ffffff;
}
.st-key-perfil_alto button:hover {
    border-color: #e34948; color: #df3130; background-color: rgba(227, 73, 72, 0.10);
}
[data-testid="stNumberInput"] label p { font-size: 0.82rem !important; color: #52514e !important; }
hr { margin: 1.4rem 0 !important; border-color: #e1e0d9 !important; }
.subtitulo { font-size: 0.95rem; color: #52514e; margin: 0 0 0.4rem 0; }
.bloco {
    border-left: 3px solid %(borda)s; background: #f6f6f3;
    padding: 0.8rem 1.05rem; margin: 0.35rem 0 0.7rem 0; border-radius: 2px;
}
.bloco-rotulo {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b6a66;
}
.bloco-valor { font-size: 1.2rem; font-weight: 600; color: #0b0b0b; margin-top: 0.15rem; }
.bloco-texto { font-size: 0.92rem; color: #2b2a28; margin-top: 0.2rem; line-height: 1.5; }
</style>
"""
st.markdown(CSS % {"borda": "#c3c2b7"}, unsafe_allow_html=True)


@st.cache_data
def carregar_referencias():
    """Recalcula o split de treino (mesmos parametros do treino oficial) para
    obter a mediana do treino (valor inicial dos campos) e perfis reais de
    usuarios tipicos de cada classe de risco."""
    df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")
    X = df[FEATURES]
    y = df["classificacao_risco"]

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_enc = le.transform(y)

    X_train, _, _, _ = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    medianas_treino = X_train.median().to_dict()

    # perfil "tipico" de cada classe = usuario real mais proximo da mediana
    # (padronizada) daquela classe, entre os que o modelo classifica corretamente
    pipeline, metadata = _carregar()
    label_para_nome = {v: k for k, v in metadata["classes"]["mapeamento_label_encoder"].items()}
    previsto = pd.Series(
        [label_para_nome[int(p)] for p in pipeline.predict(X)], index=df.index
    )
    acertou = previsto == df["classificacao_risco"]

    medias, desvios = X.mean(), X.std().replace(0, 1)
    X_padronizado = (X - medias) / desvios

    perfis, user_ids = {}, {}
    for classe in CLASS_ORDER:
        idx_classe = df.index[(df["classificacao_risco"] == classe) & acertou]
        mediana_classe = X_padronizado.loc[idx_classe].median()
        distancias = ((X_padronizado.loc[idx_classe] - mediana_classe) ** 2).sum(axis=1)
        usuario_tipico = distancias.idxmin()
        perfis[classe] = df.loc[usuario_tipico, FEATURES].to_dict()
        user_ids[classe] = int(df.loc[usuario_tipico, "UserID"])

    return medianas_treino, perfis, user_ids


medianas_treino, perfis_exemplo, user_ids_exemplo = carregar_referencias()

if "inicializado" not in st.session_state:
    for feat in FEATURES:
        st.session_state[feat] = valor_tipado(feat, medianas_treino[feat])
    st.session_state["inicializado"] = True


def aplicar_perfil(perfil):
    for feat in FEATURES:
        st.session_state[feat] = valor_tipado(feat, perfil[feat])


def bloco(rotulo, valor=None, texto=None, cor="#c3c2b7"):
    partes = [f'<div class="bloco" style="border-left-color:{cor};">',
              f'<div class="bloco-rotulo">{html.escape(rotulo)}</div>']
    if valor is not None:
        partes.append(f'<div class="bloco-valor">{html.escape(str(valor))}</div>')
    if texto is not None:
        partes.append(f'<div class="bloco-texto">{html.escape(str(texto))}</div>')
    partes.append("</div>")
    st.markdown("".join(partes), unsafe_allow_html=True)


def grafico_probabilidades(probabilidades):
    """Barras horizontais das 3 probabilidades, com trilha de fundo e valor na ponta."""
    fig, ax = plt.subplots(figsize=(7.2, 1.75), dpi=200)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    posicoes = range(len(CLASS_ORDER))
    valores = [probabilidades[c] for c in CLASS_ORDER]

    espessura = 0.3  # marca fina: a sobra da faixa fica como ar
    ax.barh(posicoes, [1] * len(CLASS_ORDER), height=espessura, color=TRILHA, zorder=1)
    ax.barh(
        posicoes, valores, height=espessura,
        color=[CORES_RISCO[c] for c in CLASS_ORDER], zorder=2,
    )

    for pos, valor in zip(posicoes, valores):
        ax.text(
            min(valor + 0.015, 1.0), pos, f"{valor * 100:.1f}%",
            va="center", ha="left", fontsize=9, color=TINTA_SECUNDARIA, zorder=3,
        )

    ax.set_yticks(list(posicoes))
    ax.set_yticklabels([NOME_EXIBICAO[c] for c in CLASS_ORDER], fontsize=9.5, color=TINTA_PRIMARIA)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    return fig


st.title("Classificação de risco comportamental em apostas online")
st.markdown(
    '<p class="subtitulo">Demonstração do modelo de classificação desenvolvido no TCC '
    "(XGBoost ajustado sobre 12 variáveis comportamentais).</p>",
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Perfis de exemplo")
st.caption(
    "Cada botão preenche os campos com os valores de um usuário real do dataset, o mais próximo "
    f"do perfil mediano da sua classe: baixo risco (usuário {user_ids_exemplo['Baixo risco']}), "
    f"médio risco (usuário {user_ids_exemplo['Medio risco']}) e "
    f"alto risco (usuário {user_ids_exemplo['Alto risco']})."
)
colunas_perfis = st.columns(3)
for coluna, classe, rotulo, chave in zip(
    colunas_perfis,
    CLASS_ORDER,
    ("Perfil de baixo risco", "Perfil de médio risco", "Perfil de alto risco"),
    ("perfil_baixo", "perfil_medio", "perfil_alto"),  # gera a classe CSS st-key-<chave>
):
    with coluna:
        st.button(
            rotulo,
            key=chave,
            width="stretch",
            on_click=partial(aplicar_perfil, perfis_exemplo[classe]),
        )

st.divider()

st.subheader("Variáveis comportamentais")
st.caption(
    "Os valores iniciais são a mediana de cada variável no conjunto de treino. Como cada mediana "
    "é calculada de forma independente, esse conjunto inicial não corresponde a um usuário real e "
    "pode gerar avisos de coerência — use os perfis de exemplo acima para ver um caso consistente."
)
colunas = st.columns(3)
for i, feat in enumerate(FEATURES):
    formato, passo = formato_campo(feat)
    with colunas[i % 3]:
        st.number_input(
            ROTULOS[feat],
            key=feat,
            help=AJUDA[feat],
            format=formato,
            step=passo,
        )

st.divider()
classificar = st.button("Classificar")

if classificar:
    entrada = {feat: st.session_state[feat] for feat in FEATURES}
    try:
        resultado = prever_risco(entrada)
    except ValueError as e:
        bloco("Entrada inválida", texto=str(e), cor=CORES_RISCO["Alto risco"])
    else:
        st.subheader("Resultado")

        classe = resultado["classe_prevista"]
        bloco("Classe prevista", valor=NOME_EXIBICAO[classe], cor=CORES_RISCO[classe])

        st.markdown('<div class="bloco-rotulo">Probabilidades por classe</div>', unsafe_allow_html=True)
        st.pyplot(grafico_probabilidades(resultado["probabilidades"]), width="content")

        intervencao = resultado["intervencao"]
        acoes = " · ".join(a["tipo"] for a in intervencao["acoes"])
        bloco(
            "Intervenção recomendada",
            texto=f"{intervencao['descricao']}\n\n{intervencao['codigo']} — {acoes}",
        )

        if resultado["avisos"]:
            for aviso in resultado["avisos"]:
                bloco("Aviso de validação", texto=aviso, cor=CORES_RISCO["Medio risco"])
