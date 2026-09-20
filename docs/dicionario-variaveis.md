# Dicionário de Variáveis

Especificação técnica das doze variáveis comportamentais que compõem a entrada do modelo de classificação de risco. Documento de referência para integração e reprodução.

| | |
|---|---|
| **Variáveis** | 12 (7 base, 5 derivadas) |
| **Base de referência** | bwin — Gray, LaPlante e Shaffer (2012) |
| **Usuários** | 4.113 |
| **Valores ausentes** | 0 |

---

## Antes de integrar

O modelo recebe doze números já calculados. Ele não processa registros brutos de apostas — a agregação por usuário é responsabilidade de quem integra.

### A ordem das variáveis é obrigatória

O modelo espera as doze colunas na ordem exata registrada em `metadata.json`. Colunas trocadas de posição **não produzem erro** — produzem uma classificação incorreta sem nenhum aviso. Reordene sempre pela lista dos metadados antes de chamar a predição.

### Valores ausentes são preenchidos silenciosamente

O pipeline embutido substitui qualquer valor ausente pela mediana calculada no treino, sem emitir aviso. Uma predição feita com três das doze variáveis ausentes retorna normalmente, com aparência de confiabilidade idêntica a uma predição completa. Verifique a integridade da entrada antes de chamar o modelo.

### Valores absolutos não transferem entre populações

As faixas documentadas aqui refletem a base bwin entre 2005 e 2007. Uma plataforma com perfil de movimentação diferente produzirá classificações deslocadas se enviar valores absolutos. A conversão de moeda não resolve o problema, porque o modelo aprendeu posições relativas dentro daquela população, não patamares monetários.

---

## Sobre a distribuição dos dados

As variáveis monetárias e de contagem apresentam assimetria acentuada à direita. A média é várias vezes maior que a mediana, indicando que uma minoria de usuários de volume muito elevado desloca todo o valor médio.

| Variável | Mediana | Média | Razão |
|---|---:|---:|---:|
| `total_apostado` | 1.416,01 | 66.016,88 | **46,6×** |
| `total_perdas` | 143,03 | 3.130,16 | **21,9×** |
| `numero_total_apostas` | 725 | 10.729,89 | **14,8×** |
| `desvio_padrao_apostado` | 45,63 | 534,85 | 11,7× |
| `media_diaria_apostada` | 33,53 | 288,17 | 8,6× |
| `media_apostas_por_produto` | 252,25 | 2.150,36 | 8,5× |
| `intensidade_apostas_por_dia_ativo` | 11,98 | 50,34 | 4,2× |
| `total_dias_ativos` | 62 | 159,75 | 2,6× |

O caso mais extremo é `total_apostado`: metade dos usuários movimentou menos de 1.416 no histórico completo, enquanto o máximo observado ultrapassa doze milhões. A média de 66 mil não descreve nenhum usuário típico.

> **Implicação prática**
>
> Essa assimetria é o argumento técnico central para normalizar por percentil em vez de usar valores absolutos ao aplicar o modelo a outra população. Em distribuições com cauda longa, pequenas diferenças de escala entre populações deslocam uma fração grande dos usuários para faixas incorretas. O percentil é invariante a esse tipo de diferença.

---

# Variáveis base

Agregadas diretamente dos registros diários do dataset II.

---

## `total_dias_ativos`

**Tipo:** inteiro · **Unidade:** dias

Número de dias distintos em que o usuário registrou pelo menos uma aposta, ao longo de todo o histórico disponível.

```
nunique(Date) agrupado por UserID
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 1 | 14 | **62** | 159,8 | 201 | 2.356 |

**Casos especiais.** O mínimo é estruturalmente 1, já que o usuário só aparece na base se apostou ao menos um dia. **162 usuários (3,9%) têm exatamente um dia ativo** — nesses casos, `media_diaria_apostada` iguala `total_apostado`, e `desvio_padrao_apostado`, `tendencia_crescimento_apostado` e `concentracao_apostas_dias_ativos` são forçados a zero por indefinição estatística.

---

## `total_apostado`

**Tipo:** decimal · **Unidade:** monetária

Soma de todo o valor apostado pelo usuário, considerando todos os produtos e todos os dias do histórico. Corresponde ao *turnover* acumulado, não ao saldo movimentado.

```
sum(Turnover) agrupado por UserID
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 0 | 167,26 | **1.416,01** | 66.016,88 | 19.863,43 | 12.123.136,88 |

**Casos especiais.** **96 usuários com total apostado igual a zero.** Como esta variável é o denominador de `taxa_perda_sobre_apostado`, a divisão por zero é tratada substituindo o denominador por ausente e preenchendo o resultado com zero. É a variável de maior amplitude e maior assimetria de toda a base.

---

## `total_perdas`

**Tipo:** decimal · **Unidade:** monetária

Resultado líquido acumulado da operadora sobre o usuário. Valor positivo indica que o usuário perdeu no agregado; **valor negativo indica que o usuário ficou no lucro**.

```
sum(Hold) agrupado por UserID
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| **−35.851,11** | 25,00 | **143,03** | 3.130,16 | 1.564,10 | 181.056,00 |

**Casos especiais.** **408 usuários (9,9%) apresentam valor negativo.** Apesar do nome, a variável não representa apenas perdas: admite valores negativos quando o usuário ganhou mais do que perdeu no acumulado. Isso é dado válido da base original, não erro de processamento. Quem integrar deve enviar o resultado líquido, e não a soma apenas das apostas perdidas.

---

## `numero_total_apostas`

**Tipo:** contagem · **Unidade:** apostas

Quantidade total de apostas individuais realizadas pelo usuário em todo o histórico, somando todos os produtos.

```
sum(NumberofBets) agrupado por UserID
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 1 | 76 | **725** | 10.729,9 | 5.402 | 2.371.197 |

**Casos especiais.** Sem risco de indefinição. Serve de numerador em duas variáveis derivadas: `intensidade_apostas_por_dia_ativo` e `media_apostas_por_produto`.

---

## `variedade_produtos`

**Tipo:** contagem · **Unidade:** produtos distintos

Número de tipos distintos de produto em que o usuário apostou — por exemplo apostas esportivas de cota fixa, apostas ao vivo, cassino e pôquer.

```
nunique(ProductType) agrupado por UserID
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | **3** | 3,42 | 5 | 14 |

**Casos especiais.** **687 usuários (16,7%) jogaram em apenas um tipo de produto.** Nesses casos `media_apostas_por_produto` iguala `numero_total_apostas`, perdendo o efeito de diluição que a variável pretende medir. O mínimo estrutural de 1 elimina risco de divisão por zero.

---

## `media_diaria_apostada`

**Tipo:** decimal · **Unidade:** monetária por dia

Valor médio apostado por dia ativo. A agregação é feita primeiro somando o valor apostado dentro de cada dia, e só então calculando a média entre os dias.

```
mean( sum(Turnover) agrupado por UserID + Date )
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,89 | **33,53** | 288,17 | 173,91 | 41.562,40 |

**Casos especiais.** Para usuários com um único dia ativo, o valor é idêntico ao `total_apostado`, já que se trata da média de uma única observação. A ordem da agregação importa: somar por dia antes de tirar a média produz resultado diferente de tirar a média direta dos registros individuais, porque um dia pode conter apostas em vários produtos.

---

## `desvio_padrao_apostado`

**Tipo:** decimal · **Unidade:** monetária por dia

Dispersão do valor apostado entre os dias ativos. Mede a regularidade do comportamento: valores baixos indicam apostas de montante constante, valores altos indicam alternância entre dias de baixo e alto volume.

```
std( sum(Turnover) agrupado por UserID + Date, ddof=1 )
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 0 | 8,65 | **45,63** | 534,85 | 285,20 | 65.627,38 |

**Casos especiais.** O desvio padrão amostral é indefinido com uma única observação. **245 usuários (6,0%) apresentam valor exatamente zero**, reunindo dois casos distintos sob o mesmo código: usuários com um único dia ativo, em que o valor é indefinido e foi forçado a zero, e usuários com volume diário genuinamente constante. A distinção entre os dois não é recuperável a partir desta variável isoladamente.

---

# Variáveis derivadas

Construídas a partir das variáveis base.

---

## `taxa_perda_sobre_apostado`

**Tipo:** decimal · **Unidade:** proporção (adimensional)

Proporção do valor movimentado que se converteu em perda líquida. Distingue quem aposta muito e recupera de quem aposta muito e perde de forma consistente.

```
total_perdas / total_apostado
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| **−9,457** | 0,043 | **0,119** | 0,196 | 0,286 | 1,000 |

**Casos especiais.** **A faixa válida vai de −9,457 a 1,000** — qualquer validação de entrada precisa aceitar valores negativos. São 407 usuários com taxa negativa, herdada do resultado líquido positivo. Os 96 usuários com valor apostado zero geram divisão indefinida, tratada como zero. Nenhum caso excede 1,0, o que confirma consistência entre as duas variáveis de origem.

---

## `intensidade_apostas_por_dia_ativo`

**Tipo:** decimal · **Unidade:** apostas por dia

Ritmo de apostas nos dias em que o usuário esteve ativo. Separa volume de intensidade: mil apostas distribuídas em trezentos dias descrevem comportamento distinto de mil apostas concentradas em dez dias.

```
numero_total_apostas / total_dias_ativos
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 0,5 | 3,30 | **11,98** | 50,34 | 41,43 | 1.512,08 |

**Casos especiais.** Sem risco estrutural de divisão por zero, já que o denominador é sempre maior ou igual a 1 por construção da base.

---

## `tendencia_crescimento_apostado`

**Tipo:** decimal · **Unidade:** monetária por dia

Inclinação da reta de regressão do valor apostado diário contra o tempo. Valor positivo indica escalada do montante ao longo do histórico; negativo indica redução.

```
linregress( dias_desde_primeira_aposta, turnover_diario ).slope
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| **−5.313,52** | −0,088 | **−0,0006** | 1,876 | 0,055 | 6.780,00 |

**Casos especiais.** Forçada a zero quando há menos de dois dias ativos ou quando todas as apostas caem na mesma data, totalizando **245 usuários (6,0%) com valor exatamente zero**. A distribuição é peculiar: metade central extremamente concentrada em torno de zero, entre −0,088 e 0,055, com caudas que alcançam milhares em ambas as direções. Poucos dias ativos muito espaçados tornam a inclinação altamente sensível a observações isoladas. É a variável de menor importância no modelo final, e essa instabilidade explica em parte o resultado.

---

## `concentracao_apostas_dias_ativos`

**Tipo:** decimal · **Unidade:** índice de Gini (adimensional)

Coeficiente de Gini aplicado à distribuição do valor apostado entre os dias ativos. Mede se a atividade está espalhada de forma regular ou concentrada em poucos dias de intensidade elevada.

```
gini( turnover_diario )
0 = distribuição uniforme · 1 = concentração total
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,501 | **0,619** | 0,599 | 0,743 | 0,998 |

**Casos especiais.** Forçada a zero quando há um único dia ativo ou quando a soma do valor apostado é nula, situações em que a fórmula é indefinida — os mesmos **245 usuários (6,0%)** das duas variáveis anteriores. Note que zero aqui é ambíguo: representa tanto distribuição perfeitamente uniforme quanto indefinição. É a única variável cuja média e mediana praticamente coincidem, indicando distribuição simétrica.

---

## `media_apostas_por_produto`

**Tipo:** decimal · **Unidade:** apostas por produto

Número médio de apostas por tipo de produto acessado. Distingue o usuário que concentra atividade em poucos produtos daquele que distribui o mesmo volume por várias modalidades.

```
numero_total_apostas / variedade_produtos
```

| Mínimo | Q1 | Mediana | Média | Q3 | Máximo |
|---:|---:|---:|---:|---:|---:|
| 1 | 38 | **252,25** | 2.150,36 | 1.335 | 474.239,40 |

**Casos especiais.** Sem risco de divisão por zero, pois o denominador é sempre maior ou igual a 1. Para os 687 usuários de produto único, o valor coincide com `numero_total_apostas`.

---

# Síntese dos casos especiais

Situações que exigem tratamento explícito por quem for calcular as variáveis. Todas já estão resolvidas no pipeline de referência; a tabela serve para reprodução em outro ambiente.

| Situação | Usuários | Tratamento |
|---|---:|---|
| Um único dia ativo | 162 · 3,9% | Desvio padrão, tendência e concentração forçados a zero |
| Valor apostado igual a zero | 96 · 2,3% | Taxa de perda definida como zero após substituir o denominador |
| **Resultado líquido negativo** | **408 · 9,9%** | **Mantido como está — valor válido, não é erro** |
| Produto único | 687 · 16,7% | Sem tratamento — média por produto iguala o total de apostas |
| Estatística indefinida | 245 · 6,0% | Três variáveis forçadas a zero por indefinição |

---

# Regras de validação recomendadas

Verificações a implementar antes de submeter dados ao modelo, em ordem de criticidade.

**1. Conferir presença e ordem das doze variáveis.** Comparar com a lista de `metadata.json` e reordenar se necessário. Ausência deve gerar erro explícito, não imputação silenciosa.

**2. Rejeitar valores negativos onde são impossíveis.** Dias ativos, valor apostado, número de apostas, variedade de produtos, média diária, desvio padrão, intensidade, concentração e média por produto não admitem valores negativos. Três variáveis aceitam: `total_perdas`, quando o usuário ficou no lucro; `taxa_perda_sobre_apostado`, pela mesma razão; e `tendencia_crescimento_apostado`, em que valor negativo indica redução do montante apostado ao longo do tempo — situação comum, observada em mais da metade da base.

**3. Verificar coerência entre variáveis dependentes.** A intensidade deve corresponder ao número de apostas dividido pelos dias ativos, e a média por produto ao número de apostas dividido pela variedade. Divergência indica erro de cálculo na origem.

**4. Limitar a concentração ao intervalo válido.** O coeficiente de Gini pertence obrigatoriamente ao intervalo entre 0 e 1. Valores fora indicam erro de implementação da fórmula.

**5. Sinalizar entradas fora da faixa de referência.** Valores muito acima do máximo observado não impedem a predição, mas o modelo satura na folha mais extrema em vez de extrapolar. A predição permanece válida, embora a probabilidade não diferencie graus além do limite conhecido.

---

## Referência da base

GRAY, Heather M.; LAPLANTE, Debi A.; SHAFFER, Howard J. Behavioral characteristics of Internet gamblers who trigger corporate responsible gambling interventions. **Psychology of Addictive Behaviors**, v. 26, n. 3, p. 527-535, 2012.

Dados disponíveis publicamente pelo Transparency Project, Division on Addiction, Cambridge Health Alliance — hospital de ensino da Harvard Medical School.
