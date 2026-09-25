# Classificação de risco comportamental em apostas online

**[Demonstração ao vivo](https://gambling-risk-classifier.onrender.com)** — a primeira visita pode levar até um minuto para carregar, porque o serviço gratuito hiberna após um período de inatividade.

Modelo de aprendizado de máquina que classifica apostadores em três níveis de risco de comportamento compulsivo — baixo, médio e alto — a partir de doze variáveis comportamentais derivadas do histórico de apostas.

A cada nível corresponde uma intervenção proporcional, do monitoramento padrão ao encaminhamento para equipe de jogo responsável.

| | |
|---|---|
| **Modelo** | XGBoost com hiperparâmetros ajustados |
| **Variáveis** | 12 comportamentais, sem dados demográficos |
| **AUC macro** | 0,915 no teste retido · 0,917 ± 0,010 em validação cruzada (10 divisões) |
| **Recall macro** | 0,799 |
| **Base de treino** | bwin — Gray, LaPlante e Shaffer (2012), 4.113 apostadores |

---

## O que há aqui

Uma API REST para integração, um painel web de demonstração, o modelo treinado e serializado, o pipeline completo de preparação dos dados, e o registro dos quatorze experimentos que levaram à configuração final.

A documentação em `docs/` cobre a especificação das variáveis de entrada, as limitações do modelo e o percurso de desenvolvimento.

---

## Instalação

Requer Python 3.12 ou superior.

```bash
git clone https://github.com/luccaschneider/gambling-risk-classifier.git
cd gambling-risk-classifier
pip install -r requirements.txt
```

## Execução

```bash
uvicorn src.api:app --reload --port 8000
```

Dois endereços ficam disponíveis:

- `http://127.0.0.1:8000` — painel de demonstração
- `http://127.0.0.1:8000/docs` — documentação interativa da API

O painel traz três perfis de exemplo extraídos do dataset, que preenchem os campos com valores de apostadores reais de cada classe de risco.

---

## Uso da API

**Classificar um apostador**

```http
POST /classificar
Content-Type: application/json
```

```json
{
  "total_dias_ativos": 120,
  "total_apostado": 15000,
  "total_perdas": 2200,
  "numero_total_apostas": 3400,
  "variedade_produtos": 4,
  "media_diaria_apostada": 125,
  "desvio_padrao_apostado": 300,
  "taxa_perda_sobre_apostado": 0.1467,
  "intensidade_apostas_por_dia_ativo": 28.33,
  "tendencia_crescimento_apostado": 0.5,
  "concentracao_apostas_dias_ativos": 0.6,
  "media_apostas_por_produto": 850
}
```

O campo `usuario_externo` é opcional e serve para rastrear a decisão até o cadastro da operadora.

```json
{
  "classe_prevista": "Alto risco",
  "probabilidades": {
    "Baixo risco": 0.0063,
    "Medio risco": 0.1008,
    "Alto risco": 0.8930
  },
  "indice_risco": 0.9427,
  "intervencao": {
    "codigo": "SUSPENSAO_TEMPORARIA",
    "nivel": "Alto risco",
    "acoes": [
      {
        "tipo": "SUSPENDER_CONTA",
        "alvo": "conta",
        "carater": "temporario",
        "bloqueia_conta": true,
        "parametros": { "duracao_dias": 30, "reversivel": true }
      },
      {
        "tipo": "ENCAMINHAR_SUPORTE",
        "alvo": "equipe_interna",
        "carater": "atendimento",
        "bloqueia_conta": false,
        "parametros": { "canal": "suporte_jogo_responsavel", "prazo_contato_horas": 48 }
      }
    ],
    "descricao": "Suspensão temporária da conta e encaminhamento para canal de suporte especializado em jogo responsável."
  },
  "avisos": [],
  "registro_id": 12
}
```

A intervenção vem estruturada para ser executada sem interpretar texto: o `codigo` identifica o nível de resposta e cada ação traz `tipo`, `alvo` e se bloqueia a conta. Os valores em `parametros` são de referência — prazos e percentuais devem ser definidos pela operadora conforme sua política de jogo responsável, não são recomendação clínica.

O `indice_risco` vai de 0 a 1 e posiciona o usuário na escala contínua entre os três níveis. É derivado das probabilidades, não uma saída direta do modelo.

**Outros endpoints**

`GET /saude` — confirma que o modelo está carregado
`GET /variaveis` — lista as variáveis esperadas, na ordem, com as faixas observadas
`GET /registros` — últimas classificações registradas, para auditoria; aceita `?limite=N`

**Registro auditável**

Cada chamada ao `POST /classificar` é gravada em SQLite com data e hora em UTC, as doze entradas, o resultado, o código da intervenção, os avisos e a versão do modelo. Nesta demonstração o banco fica em arquivo temporário e guarda as 500 entradas mais recentes — o plano gratuito do Render não tem disco persistente, então o registro se perde quando o serviço reinicia ou hiberna. Em uso real, seria uma tabela no banco da operadora.

---

## Uso direto em Python

```python
import sys
sys.path.insert(0, "src")
from predicao import prever_risco

resultado = prever_risco({
    "total_dias_ativos": 120,
    "total_apostado": 15000,
    # ... as doze variáveis
})

print(resultado["classe_prevista"])
print(resultado["probabilidades"])
```

A função verifica a presença das doze variáveis, valida os domínios, confere a coerência entre variáveis derivadas e sinaliza valores fora da faixa de treino. Carregar o arquivo serializado diretamente contorna essas verificações.

---

## Estrutura

```
modelo/         Modelo serializado e metadados
src/            API, função de predição e app Streamlit
static/         Painel web
pipeline/       Preparação dos dados e treino, em ordem de execução
experimentos/   Scripts dos quatorze experimentos
resultados/     Métricas e figuras gerados
testes/         Validação do modelo, da função e da API
docs/           Documentação
dados/          Arquivos da base (não versionados)
```

---

## Reproduzir o treinamento

Os dados não acompanham o repositório. São distribuídos pelo Transparency Project sob condições próprias, e os arquivos intermediários somam cerca de 132 MB.

Para reproduzir:

1. Obtenha os três arquivos brutos em [thetransparencyproject.org](http://www.thetransparencyproject.org/download_index.php), no estudo *Behavioral Characteristics of Internet Gamblers Who Trigger Corporate Responsible Gambling Interventions*.

2. Coloque-os em `dados/`.

3. Execute o pipeline na ordem:

```bash
python pipeline/01_merge_bwin.py
python pipeline/02_aggregate_bwin.py
python pipeline/03_engineer_features.py
python pipeline/04_train_final_model.py
```

O modelo resultante substitui o que está em `modelo/`. Para confirmar que reproduz o original:

```bash
python testes/validate_serialization.py
```

---

## Documentação

**[Dicionário de variáveis](docs/dicionario-variaveis.md)** — definição operacional, regra de cálculo, unidade e distribuição observada de cada uma das doze variáveis. Leitura necessária antes de integrar.

**[Guia de uso e limitações](docs/guia-uso-limitacoes.md)** — alcance do modelo, restrições de transferência entre populações, comportamento em casos-limite e considerações éticas.

**[Relatório de experimentos](docs/relatorio-experimentos.html)** — registro cronológico das quatorze etapas de desenvolvimento, com resultados e justificativa de cada decisão.

---

## Antes de aplicar

Três pontos do guia merecem destaque.

**Valores absolutos não transferem entre populações.** O modelo aprendeu posições relativas dentro da base bwin, não patamares monetários. Aplicá-lo a uma plataforma com perfil de movimentação diferente produz classificações deslocadas, e converter a moeda não resolve. A mitigação recomendada é normalizar cada variável por percentil dentro da população local.

**A classificação não é diagnóstico.** Indica padrão comportamental compatível com risco. O diagnóstico de transtorno do jogo é ato clínico.

**O erro é assimétrico por decisão de projeto.** Entre os casos reais de médio risco, 37,7% são classificados como alto risco, enquanto o erro inverso é raro. Técnicas de balanceamento foram testadas e descartadas: reduziriam esse desvio ao custo de onze a treze pontos no recall da classe de alto risco, onde deixar passar tem consequência maior.

---

## Desempenho

Validação cruzada estratificada em dez divisões:

| | AUC | F1 | Recall |
|---|---:|---:|---:|
| **Macro** | 0,917 ± 0,010 | 0,796 ± 0,015 | 0,799 ± 0,016 |
| Baixo risco | 0,984 | 0,914 | 0,959 |
| Médio risco | 0,881 | 0,655 | 0,609 |
| Alto risco | 0,886 | 0,819 | 0,828 |

No conjunto de teste retido, com 20% dos usuários, o AUC macro foi de 0,915. Como as decisões dos experimentos foram tomadas comparando resultados nesse mesmo conjunto, os dois números devem ser lidos juntos: a convergência entre eles indica que as escolhas não se ajustaram a uma partição específica.

A classe de médio risco tem desempenho inferior por razão metodológica: enquanto o alto risco corresponde a um evento concreto — a marcação pelo programa de jogo responsável da operadora —, a fronteira entre baixo e médio resulta de um corte estatístico em distribuição contínua. Usuários imediatamente acima e abaixo do limiar são comportamentalmente quase indistinguíveis.

---

## Base de dados

GRAY, Heather M.; LAPLANTE, Debi A.; SHAFFER, Howard J. Behavioral characteristics of Internet gamblers who trigger corporate responsible gambling interventions. **Psychology of Addictive Behaviors**, v. 26, n. 3, p. 527-535, 2012.

Dados disponibilizados publicamente pelo Transparency Project, Division on Addiction, Cambridge Health Alliance — hospital de ensino da Harvard Medical School.

---

## Licença

MIT
