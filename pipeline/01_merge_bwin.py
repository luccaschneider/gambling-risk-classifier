import pandas as pd
from pathlib import Path

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
METRICAS = RAIZ / "resultados" / "metricas"

demo = pd.read_csv(DADOS / "Raw Datset I.Demographics_Gray_LaPlante_PAB_2012.dat.txt", sep="\t")
daily = pd.read_csv(DADOS / "Raw Datset II.Daily aggregates_Gray_LaPlante_PAB_2012.dat.txt", sep="\t")
rg = pd.read_csv(DADOS / "Raw Datset III.Responsible gambling details_Gray_LaPlante_PAB_2012.dat.txt", sep="\t")

# normalize the User ID column name across the three files
for df in (demo, daily, rg):
    df.rename(columns={c: "UserID" for c in df.columns if c.upper() == "USERID"}, inplace=True)

print("Demographics:", demo.shape, list(demo.columns))
print("Daily aggregates:", daily.shape, list(daily.columns))
print("Responsible gambling:", rg.shape, list(rg.columns))

combined = daily.merge(demo, on="UserID", how="left").merge(rg, on="UserID", how="left")

print("\nCombined shape:", combined.shape)
print("\nHead of combined dataset:")
print(combined.head(10).to_string())

combined.to_csv(DADOS / "combined_bwin_dataset.csv", index=False)
print("\nSaved to combined_bwin_dataset.csv")
