import pandas as pd
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

demo = pd.read_csv(DADOS / "Raw Datset I.Demographics_Gray_LaPlante_PAB_2012.dat.txt", sep="\t")
daily = pd.read_csv(DADOS / "Raw Datset II.Daily aggregates_Gray_LaPlante_PAB_2012.dat.txt", sep="\t", low_memory=False)
rg = pd.read_csv(DADOS / "Raw Datset III.Responsible gambling details_Gray_LaPlante_PAB_2012.dat.txt", sep="\t")

for df in (demo, daily, rg):
    df.rename(columns={c: "UserID" for c in df.columns if c.upper() == "USERID"}, inplace=True)

daily["Turnover"] = pd.to_numeric(daily["Turnover"], errors="coerce")
daily["Hold"] = pd.to_numeric(daily["Hold"], errors="coerce")
daily["NumberofBets"] = pd.to_numeric(daily["NumberofBets"], errors="coerce")

# totals apostados por usuario/dia (um usuario pode ter varios produtos no mesmo dia)
daily_totals = daily.groupby(["UserID", "Date"], as_index=False)["Turnover"].sum()
daily_stats = daily_totals.groupby("UserID")["Turnover"].agg(
    media_diaria_apostada="mean",
    desvio_padrao_apostado="std"
).reset_index()
daily_stats["desvio_padrao_apostado"] = daily_stats["desvio_padrao_apostado"].fillna(0)

agg = daily.groupby("UserID").agg(
    total_dias_ativos=("Date", "nunique"),
    total_apostado=("Turnover", "sum"),
    total_perdas=("Hold", "sum"),
    numero_total_apostas=("NumberofBets", "sum"),
    variedade_produtos=("ProductType", "nunique"),
).reset_index()

behavior = agg.merge(daily_stats, on="UserID", how="left")

# une com demograficos e RG
combined = behavior.merge(demo, on="UserID", how="left").merge(rg, on="UserID", how="left")

# flag RG: RG_case==1 (idêntico a estar presente no dataset III -> 2068 usuarios em ambos)
rg_flag = combined["RG_case"] == 1

# --- criterio composto de intensidade (apenas para usuarios SEM flag RG) ---
# cada uma das 4 variaveis comportamentais e convertida em percentil (0-100)
# dentro da propria populacao sem flag RG, e a media dos 4 percentis vira um
# "escore de intensidade" (0-100) que resume, num unico numero, o quao intenso
# e o padrao de jogo do usuario nas 4 dimensoes ao mesmo tempo (nao so o valor
# apostado, mas tambem frequencia, diversidade de produtos e volatilidade).
intensity_vars = ["total_dias_ativos", "total_apostado", "variedade_produtos", "desvio_padrao_apostado"]

sem_rg = ~rg_flag
for var in intensity_vars:
    pct_col = f"pct_{var}"
    combined.loc[sem_rg, pct_col] = combined.loc[sem_rg, var].rank(pct=True) * 100

pct_cols = [f"pct_{v}" for v in intensity_vars]
combined.loc[sem_rg, "escore_intensidade"] = combined.loc[sem_rg, pct_cols].mean(axis=1)

# limiar = mediana do escore (percentil 50) entre os usuarios sem flag RG
threshold = combined.loc[sem_rg, "escore_intensidade"].median()

def classificar(row):
    if row["RG_case"] == 1:
        return "Alto risco"
    elif row["escore_intensidade"] > threshold:
        return "Medio risco"
    else:
        return "Baixo risco"

combined["classificacao_risco"] = combined.apply(classificar, axis=1)
combined.drop(columns=pct_cols, inplace=True)

print("Criterio composto: media dos percentis de", intensity_vars)
print("Threshold (mediana do escore de intensidade, usuarios sem flag RG):", round(threshold, 2))
print("\nDistribuicao da classificacao de risco:")
print(combined["classificacao_risco"].value_counts())

print("\nShape final:", combined.shape)
print("\nHead do dataset agregado e classificado:")
cols_show = ["UserID", "total_dias_ativos", "total_apostado", "desvio_padrao_apostado",
             "variedade_produtos", "escore_intensidade", "CountryName", "Gender",
             "RG_case", "classificacao_risco"]
print(combined[cols_show].head(10).to_string())

combined.to_csv(DADOS / "bwin_dataset_agregado_classificado.csv", index=False)
print("\nSalvo em bwin_dataset_agregado_classificado.csv")
