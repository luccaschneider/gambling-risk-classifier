import numpy as np
import pandas as pd
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")

FEATURES = [
    "total_dias_ativos", "total_apostado", "total_perdas", "numero_total_apostas",
    "variedade_produtos", "media_diaria_apostada", "desvio_padrao_apostado",
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
]

# --- casos especiais reais observados no dataset -----------------------------
n_total = len(df)
n_um_dia_ativo = int((df["total_dias_ativos"] == 1).sum())
n_apostado_zero = int((df["total_apostado"] == 0).sum())
n_perdas_negativas = int((df["total_perdas"] < 0).sum())
n_produto_unico = int((df["variedade_produtos"] == 1).sum())
n_desvio_zero = int((df["desvio_padrao_apostado"] == 0).sum())
n_tendencia_zero = int((df["tendencia_crescimento_apostado"] == 0).sum())
n_concentracao_zero = int((df["concentracao_apostas_dias_ativos"] == 0).sum())
n_taxa_perda_negativa = int((df["taxa_perda_sobre_apostado"] < 0).sum())
n_taxa_perda_maior_1 = int((df["taxa_perda_sobre_apostado"] > 1).sum())

DOCS = {
    "total_dias_ativos": dict(
        tipo_dado="int",
        formula="nunique(Date) agrupado por UserID no dataset II (dias distintos com pelo menos 1 aposta)",
        unidade="dias",
        casos_especiais=(
            f"Minimo estrutural = 1 (usuario aparece no dataset II so se apostou pelo menos 1 dia). "
            f"{n_um_dia_ativo} usuarios ({n_um_dia_ativo/n_total*100:.1f}%) tem exatamente 1 dia ativo - "
            f"nesses casos, media_diaria_apostada = total_apostado e desvio_padrao_apostado / "
            f"tendencia_crescimento_apostado sao forcados a 0 (nao ha variancia nem serie temporal)."
        ),
    ),
    "total_apostado": dict(
        tipo_dado="float",
        formula="sum(Turnover) agrupado por UserID no dataset II (soma de todas as apostas, todos os produtos e dias)",
        unidade="unidade monetaria (moeda normalizada do dataset bwin)",
        casos_especiais=(
            f"{n_apostado_zero} usuarios com total_apostado = 0. Esse valor e o denominador de "
            f"taxa_perda_sobre_apostado - divisao por zero e tratada explicitamente no pipeline "
            f"(substituida por NaN e depois preenchida com 0)."
        ),
    ),
    "total_perdas": dict(
        tipo_dado="float",
        formula="sum(Hold) agrupado por UserID no dataset II (Hold = receita liquida da casa; pode ser negativo)",
        unidade="unidade monetaria (moeda normalizada do dataset bwin)",
        casos_especiais=(
            f"{n_perdas_negativas} usuarios ({n_perdas_negativas/n_total*100:.1f}%) com total_perdas negativo "
            f"(usuario ganhou mais do que apostou na soma do periodo - Hold negativo e um valor valido "
            f"no dataset original, nao e erro). Isso faz taxa_perda_sobre_apostado tambem ser negativa "
            f"nesses casos."
        ),
    ),
    "numero_total_apostas": dict(
        tipo_dado="float (contagem, sem casas decimais na pratica)",
        formula="sum(NumberofBets) agrupado por UserID no dataset II",
        unidade="contagem de apostas",
        casos_especiais="Nenhum caso de divisao por zero direta; e numerador em duas features derivadas (intensidade e media por produto).",
    ),
    "variedade_produtos": dict(
        tipo_dado="int",
        formula="nunique(ProductType) agrupado por UserID no dataset II",
        unidade="contagem de tipos de produto distintos",
        casos_especiais=(
            f"Minimo estrutural = 1. {n_produto_unico} usuarios ({n_produto_unico/n_total*100:.1f}%) jogaram "
            f"apenas 1 tipo de produto. E o denominador de media_apostas_por_produto - como o minimo e sempre "
            f">=1, nao ha risco estrutural de divisao por zero, mas o caso de 1 produto faz "
            f"media_apostas_por_produto == numero_total_apostas (sem efeito de diluicao)."
        ),
    ),
    "media_diaria_apostada": dict(
        tipo_dado="float",
        formula="media do turnover diario (Turnover somado por UserID+Date, depois media entre os dias ativos)",
        unidade="unidade monetaria por dia",
        casos_especiais="Para usuarios com 1 dia ativo, e identica ao total_apostado (media de 1 unico valor).",
    ),
    "desvio_padrao_apostado": dict(
        tipo_dado="float",
        formula="desvio padrao amostral (ddof=1) do turnover diario (mesma agregacao de media_diaria_apostada)",
        unidade="unidade monetaria por dia",
        casos_especiais=(
            f"Desvio padrao amostral com ddof=1 e indefinido (NaN) quando ha apenas 1 observacao - "
            f"tratado explicitamente no pipeline de agregacao (fillna(0)). "
            f"{n_desvio_zero} usuarios ({n_desvio_zero/n_total*100:.1f}%) tem desvio_padrao_apostado = 0 "
            f"(inclui os usuarios de 1 dia ativo e usuarios com valor diario constante)."
        ),
    ),
    "taxa_perda_sobre_apostado": dict(
        tipo_dado="float",
        formula="total_perdas / total_apostado",
        unidade="proporcao (adimensional)",
        casos_especiais=(
            f"Divisao por zero quando total_apostado = 0 ({n_apostado_zero} casos) - tratada como 0 apos "
            f"substituir o denominador por NaN e preencher o resultado com fillna(0). "
            f"{n_taxa_perda_negativa} usuarios com taxa negativa (total_perdas < 0, usuario com saldo positivo). "
            f"{n_taxa_perda_maior_1} usuarios com taxa > 1 (perdas registradas maiores que o total apostado bruto, "
            f"pode ocorrer por ajustes/estornos no Hold que nao batem diretamente com o Turnover)."
        ),
    ),
    "intensidade_apostas_por_dia_ativo": dict(
        tipo_dado="float",
        formula="numero_total_apostas / total_dias_ativos",
        unidade="apostas por dia ativo",
        casos_especiais="Sem risco estrutural de divisao por zero (total_dias_ativos >= 1 sempre, por construcao do dataset II).",
    ),
    "tendencia_crescimento_apostado": dict(
        tipo_dado="float",
        formula=(
            "inclinacao (slope) da regressao linear (scipy.stats.linregress) do turnover diario "
            "contra os dias corridos desde a primeira aposta do usuario"
        ),
        unidade="unidade monetaria por dia (variacao do valor apostado por dia corrido)",
        casos_especiais=(
            f"Forcado para 0 quando ha menos de 2 dias ativos ({n_um_dia_ativo} usuarios) ou quando todas as "
            f"apostas caem no mesmo dia calendario (x constante, regressao indefinida). "
            f"{n_tendencia_zero} usuarios ({n_tendencia_zero/n_total*100:.1f}%) tem valor exatamente 0 por essas razoes. "
            f"Valores extremos (positivos ou negativos) podem ocorrer com poucos dias ativos muito espacados no tempo, "
            f"tornando a inclinacao sensivel a outliers."
        ),
    ),
    "concentracao_apostas_dias_ativos": dict(
        tipo_dado="float",
        formula="coeficiente de Gini da distribuicao do turnover diario entre os dias ativos do usuario",
        unidade="indice de Gini (0 a 1, adimensional)",
        casos_especiais=(
            f"Forcado para 0 quando ha <=1 dia ativo ou soma de turnover = 0 (formula indefinida nesses casos). "
            f"{n_concentracao_zero} usuarios ({n_concentracao_zero/n_total*100:.1f}%) tem valor exatamente 0. "
            f"Valores proximos de 1 indicam concentracao extrema (quase todo o valor apostado em 1-2 dias)."
        ),
    ),
    "media_apostas_por_produto": dict(
        tipo_dado="float",
        formula="numero_total_apostas / variedade_produtos",
        unidade="apostas por tipo de produto",
        casos_especiais="Sem risco estrutural de divisao por zero (variedade_produtos >= 1 sempre, por construcao do dataset II).",
    ),
}

rows = []
for feat in FEATURES:
    s = df[feat]
    desc = s.describe()
    rows.append({
        "variavel": feat,
        "tipo_dado": DOCS[feat]["tipo_dado"],
        "formula": DOCS[feat]["formula"],
        "unidade": DOCS[feat]["unidade"],
        "minimo": desc["min"],
        "q1": desc["25%"],
        "mediana": desc["50%"],
        "media": desc["mean"],
        "q3": desc["75%"],
        "maximo": desc["max"],
        "valores_faltantes": int(s.isna().sum()),
        "casos_especiais": DOCS[feat]["casos_especiais"],
    })

report = pd.DataFrame(rows)
report.to_csv(METRICAS / "documentacao_variaveis_comportamentais.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(f"Dataset: {n_total} usuarios, 12 variaveis comportamentais\n")
print("=== Resumo estatistico ===")
print(report[["variavel", "tipo_dado", "minimo", "q1", "mediana", "media", "q3", "maximo", "valores_faltantes"]]
      .round(4).to_string(index=False))

print("\n=== Formulas ===")
for feat in FEATURES:
    print(f"\n{feat}:")
    print(f"  Formula: {DOCS[feat]['formula']}")
    print(f"  Unidade: {DOCS[feat]['unidade']}")

print("\n=== Casos especiais (resumo) ===")
for feat in FEATURES:
    print(f"\n{feat}:")
    print(f"  {DOCS[feat]['casos_especiais']}")

print("\nSalvo em documentacao_variaveis_comportamentais.csv")
