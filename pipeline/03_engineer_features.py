import numpy as np
import pandas as pd
from scipy.stats import linregress
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

daily = pd.read_csv(
    DADOS / "Raw Datset II.Daily aggregates_Gray_LaPlante_PAB_2012.dat.txt", sep="\t", low_memory=False
)
daily.rename(columns={c: "UserID" for c in daily.columns if c.upper() == "USERID"}, inplace=True)
daily["Turnover"] = pd.to_numeric(daily["Turnover"], errors="coerce")
daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")

# total apostado por usuario/dia (soma entre produtos no mesmo dia)
daily_totals = daily.groupby(["UserID", "Date"], as_index=False)["Turnover"].sum()
daily_totals = daily_totals.sort_values(["UserID", "Date"])


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n <= 1 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n


def trend_slope(dates, values):
    if len(values) < 2:
        return 0.0
    x = (dates - dates.min()).dt.days.to_numpy(dtype=float)
    if np.all(x == x[0]):  # todas as apostas no mesmo dia
        return 0.0
    slope = linregress(x, values).slope
    return 0.0 if np.isnan(slope) else slope


rows = []
for user_id, g in daily_totals.groupby("UserID"):
    rows.append({
        "UserID": user_id,
        "tendencia_crescimento_apostado": trend_slope(g["Date"], g["Turnover"].to_numpy()),
        "concentracao_apostas_dias_ativos": gini(g["Turnover"].to_numpy()),
    })
time_features = pd.DataFrame(rows)

# --- une com o dataset agregado existente e cria as features simples (razoes) ---
df = pd.read_csv(DADOS / "bwin_dataset_agregado_classificado.csv")
df = df.merge(time_features, on="UserID", how="left")

df["taxa_perda_sobre_apostado"] = df["total_perdas"] / df["total_apostado"].replace(0, np.nan)
df["intensidade_apostas_por_dia_ativo"] = df["numero_total_apostas"] / df["total_dias_ativos"]
df["media_apostas_por_produto"] = df["numero_total_apostas"] / df["variedade_produtos"]

for col in ["taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo", "media_apostas_por_produto"]:
    df[col] = df[col].fillna(0)

new_features = [
    "taxa_perda_sobre_apostado", "intensidade_apostas_por_dia_ativo",
    "tendencia_crescimento_apostado", "concentracao_apostas_dias_ativos",
    "media_apostas_por_produto",
]
print("Resumo das novas variaveis:")
print(df[new_features].describe().round(4).to_string())

df.to_csv(DADOS / "bwin_dataset_features_avancadas.csv", index=False)
print("\nSalvo em bwin_dataset_features_avancadas.csv")
