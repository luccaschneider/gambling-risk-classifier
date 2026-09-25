"""
Modulo de predicao do modelo oficial de risco (XGBoost tunado, 12 variaveis
comportamentais, sem balanceamento). Carrega o pipeline e os metadados uma
unica vez (na primeira chamada) e expõe prever_risco().
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from intervencoes import intervencao_para

_MODEL_DIR = Path(__file__).resolve().parents[1] / "modelo"
_MODEL_PATH = _MODEL_DIR / "xgboost_tunado_comportamental.joblib"
_METADATA_PATH = _MODEL_DIR / "metadata.json"

_pipeline = None
_metadata = None

# tendencia_crescimento_apostado e a inclinacao de uma regressao linear: negativo
# significa apostas decrescentes ao longo do tempo (mediana do treino e negativa)
_VARIAVEIS_PODEM_SER_NEGATIVAS = {
    "total_perdas",
    "taxa_perda_sobre_apostado",
    "tendencia_crescimento_apostado",
}
_TOLERANCIA_COERENCIA = 0.01  # 1% de divergencia relativa tolerada

# pesos do indice de risco: posicao continua de 0 a 1 na escala dos tres niveis
_PESO_NIVEL = {"Baixo risco": 0.0, "Medio risco": 0.5, "Alto risco": 1.0}


def indice_de_risco(probabilidades: dict) -> float:
    """Media das probabilidades ponderada pelo nivel, de 0 a 1.

    E um valor derivado da saida do modelo, nao uma saida direta dele: serve
    para posicionar o usuario na escala continua entre os tres niveis.
    """
    return sum(probabilidades.get(nivel, 0.0) * peso for nivel, peso in _PESO_NIVEL.items())


def _carregar():
    """Carrega o pipeline e os metadados uma unica vez (lazy singleton)."""
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load(_MODEL_PATH)
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    return _pipeline, _metadata


def prever_risco(dados: dict) -> dict:
    """
    Classifica o risco de um usuario a partir das 12 variaveis comportamentais.

    Parametros
    ----------
    dados : dict
        Dicionario com as 12 variaveis comportamentais do usuario (chaves =
        nomes das colunas usadas no treino do modelo).

    Retorna
    -------
    dict com as chaves:
        classe_prevista : str  ("Baixo risco" / "Medio risco" / "Alto risco")
        probabilidades  : dict {classe: probabilidade}
        indice_risco    : float de 0 a 1, posicao na escala continua dos tres
                          niveis — derivado das probabilidades, nao saida direta
        intervencao     : dict estruturado (codigo, acoes, descricao) para a classe
        avisos          : list[str], alertas nao bloqueantes gerados na validacao

    Levanta ValueError se:
        - alguma das 12 variaveis obrigatorias estiver faltando;
        - uma variavel que so pode ser >= 0 vier negativa;
        - concentracao_apostas_dias_ativos estiver fora de [0, 1].
    """
    pipeline, metadata = _carregar()
    features = metadata["features_entrada_ordem"]
    limites_treino = metadata.get("estatisticas_treino", {})

    avisos = []

    # 1) verifica presenca de todas as 12 variaveis --------------------------
    faltando = [f for f in features if f not in dados]
    if faltando:
        raise ValueError(
            f"Variaveis faltando na entrada: {faltando}. "
            f"O modelo espera exatamente estas 12 variaveis, nesta ordem: {features}"
        )

    # 2) reordena as colunas conforme os metadados ---------------------------
    entrada = pd.DataFrame([{f: dados[f] for f in features}])[features]

    # 3) valida sinais/limites estruturais ------------------------------------
    for f in features:
        valor = entrada.at[0, f]
        if f not in _VARIAVEIS_PODEM_SER_NEGATIVAS and valor < 0:
            aceitam_negativo = ", ".join(sorted(_VARIAVEIS_PODEM_SER_NEGATIVAS))
            raise ValueError(
                f"Valor invalido: '{f}' = {valor} nao pode ser negativo. "
                f"Apenas estas variaveis aceitam valores negativos: {aceitam_negativo}."
            )

    concentracao = entrada.at[0, "concentracao_apostas_dias_ativos"]
    if not (0 <= concentracao <= 1):
        raise ValueError(
            f"Valor invalido: 'concentracao_apostas_dias_ativos' = {concentracao} "
            f"deve estar entre 0 e 1 (e um indice de Gini)."
        )

    # 4) verifica coerencia entre variaveis derivadas (aviso, nao bloqueia) --
    dias_ativos = entrada.at[0, "total_dias_ativos"]
    n_apostas = entrada.at[0, "numero_total_apostas"]
    variedade = entrada.at[0, "variedade_produtos"]

    if dias_ativos:
        intensidade_informada = entrada.at[0, "intensidade_apostas_por_dia_ativo"]
        intensidade_esperada = n_apostas / dias_ativos
        divergencia = abs(intensidade_informada - intensidade_esperada)
        if divergencia > _TOLERANCIA_COERENCIA * max(abs(intensidade_esperada), 1):
            avisos.append(
                f"intensidade_apostas_por_dia_ativo informado ({intensidade_informada:.4f}) diverge do "
                f"valor esperado numero_total_apostas/total_dias_ativos ({intensidade_esperada:.4f})."
            )

    if variedade:
        media_produto_informada = entrada.at[0, "media_apostas_por_produto"]
        media_produto_esperada = n_apostas / variedade
        divergencia = abs(media_produto_informada - media_produto_esperada)
        if divergencia > _TOLERANCIA_COERENCIA * max(abs(media_produto_esperada), 1):
            avisos.append(
                f"media_apostas_por_produto informado ({media_produto_informada:.4f}) diverge do "
                f"valor esperado numero_total_apostas/variedade_produtos ({media_produto_esperada:.4f})."
            )

    # 5) sinaliza valores acima do maximo observado no treino (nao bloqueia) -
    for f in features:
        valor = entrada.at[0, f]
        stats = limites_treino.get(f)
        if stats is not None and valor > stats["max"]:
            avisos.append(
                f"'{f}' = {valor} esta acima do maximo observado no treino ({stats['max']:.4f}) - "
                f"a predicao extrapola o range de dados visto pelo modelo."
            )

    # 6) predicao -------------------------------------------------------------
    pred = pipeline.predict(entrada)[0]
    proba = pipeline.predict_proba(entrada)[0]
    classes_modelo = list(pipeline.named_steps["clf"].classes_)
    label_para_nome = {v: k for k, v in metadata["classes"]["mapeamento_label_encoder"].items()}

    classe_prevista = label_para_nome[int(pred)]
    probabilidades = {label_para_nome[int(c)]: float(p) for c, p in zip(classes_modelo, proba)}

    return {
        "classe_prevista": classe_prevista,
        "probabilidades": probabilidades,
        "indice_risco": indice_de_risco(probabilidades),
        "intervencao": intervencao_para(classe_prevista),
        "avisos": avisos,
    }


if __name__ == "__main__":
    exemplo = {
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
    resultado = prever_risco(exemplo)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
