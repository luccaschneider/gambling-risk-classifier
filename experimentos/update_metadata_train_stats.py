"""
Adiciona ao metadata.json as estatisticas (min/max) do conjunto de TREINO,
usadas por predicao.py para sinalizar valores fora do range visto no treino.
Nao retreina o modelo, so recalcula o split (mesmos parametros) para extrair X_train.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# caminhos resolvidos a partir da raiz do projeto, para o script rodar de qualquer lugar
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
MODELO = RAIZ / "modelo"

with open(MODELO / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features_entrada_ordem"]
class_order = metadata["classes"]["ordem_risco"]

df = pd.read_csv(DADOS / "bwin_dataset_features_avancadas.csv")
X = df[features]
y = df["classificacao_risco"]

le = LabelEncoder()
le.fit(class_order)
y_enc = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=metadata["split"]["test_size"],
    random_state=metadata["split"]["random_state"], stratify=y_enc
)

estatisticas_treino = {
    f: {"min": float(X_train[f].min()), "max": float(X_train[f].max())}
    for f in features
}
metadata["estatisticas_treino"] = estatisticas_treino

with open(MODELO / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("metadata.json atualizado com estatisticas_treino (min/max por variavel):")
for f, s in estatisticas_treino.items():
    print(f"  {f}: min={s['min']:.4f}  max={s['max']:.4f}")
