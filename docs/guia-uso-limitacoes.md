# Guia de Uso e Limitações

Documento de referência sobre o alcance e as restrições do modelo de classificação de risco comportamental em plataformas de apostas online. Leitura recomendada antes de qualquer aplicação prática.

---

## O que este modelo é

Um classificador que atribui a apostadores um entre três níveis de risco — baixo, médio ou alto — a partir de doze variáveis comportamentais derivadas do histórico de apostas.

Foi desenvolvido como prova de conceito acadêmica, treinado sobre dados públicos e validado por métricas convencionais de aprendizado supervisionado. Alcançou AUC macro de 0,917 com desvio de 0,010 em validação cruzada de dez divisões, desempenho superior ao reportado pelos estudos revisados na literatura da área, que ficaram entre 0,60 e 0,79.

## O que este modelo não é

Não é um instrumento diagnóstico. A classificação indica padrão comportamental compatível com risco, não transtorno do jogo. O diagnóstico é ato clínico, feito por profissional habilitado com instrumentos validados.

Não é um produto pronto para uso em produção. Exige as adaptações descritas neste documento antes de qualquer aplicação sobre usuários reais.

Não substitui os mecanismos de jogo responsável exigidos pela regulamentação. Complementa-os, adicionando uma camada de detecção que não depende da iniciativa do usuário.

---

## Origem dos dados de treino

O modelo foi treinado sobre a base disponibilizada pelo Transparency Project, da Division on Addiction do Cambridge Health Alliance, hospital de ensino da Harvard Medical School. São registros reais de 4.113 apostadores da operadora bwin, coletados entre 2005 e 2007.

A classificação de alto risco corresponde à marcação efetuada pelo programa de jogo responsável da própria operadora. As classes de baixo e médio risco resultam de um escore composto aplicado sobre quatro variáveis comportamentais, com corte na mediana.

Essa distinção de origem entre as classes tem consequência direta no desempenho, tratada adiante na seção de limitações.

---

## A limitação central: transferência entre populações

Esta é a restrição mais importante do documento, e a que gera mais mal-entendido.

### Por que valores absolutos não transferem

O modelo não aprendeu que determinado patamar monetário indica risco. Aprendeu a posição relativa de cada usuário dentro da população específica em que foi treinado.

Na base bwin, metade dos apostadores movimentou menos de 1.416 unidades monetárias em todo o histórico. Um usuário com 20 mil estava no quartil superior, e o modelo tratou esse patamar como sinal relevante.

Uma plataforma cuja base tenha perfil de movimentação diferente produzirá classificações sistematicamente deslocadas ao enviar valores absolutos. Se o apostador mediano dessa plataforma movimenta o equivalente a 20 mil, todos eles entrarão no modelo na faixa que a bwin considerava atípica — e uma parcela grande da base será classificada em risco elevado sem que isso reflita comportamento problemático.

### A conversão de moeda não resolve

Converter valores pela cotação ajusta a unidade, não a distribuição. O problema não é que os números estejam em outra moeda; é que ocupam outra posição na curva.

Há ainda o fator temporal. Entre 2005 e hoje mudaram a renda média, os padrões de consumo e o próprio produto — a bwin daquele período era navegador, depósito por cartão e aposta antes do evento, enquanto plataformas atuais operam por aplicativo, com transferência instantânea e aposta durante a partida. A frequência de apostas por dia ativo tem outra ordem de grandeza.

### A assimetria agrava o problema

As variáveis monetárias e de contagem apresentam cauda longa à direita. Em `total_apostado`, a média é 46,6 vezes a mediana; em `total_perdas`, 21,9 vezes; em `numero_total_apostas`, 14,8 vezes.

Em distribuições assim, pequenas diferenças de escala entre populações deslocam uma fração desproporcional dos usuários para faixas incorretas. Quanto mais longa a cauda, mais sensível o modelo a diferenças de calibração.

### A mitigação recomendada

Converter cada variável em percentil dentro da própria população antes de submetê-la ao modelo, em vez de enviar valores absolutos.

Um usuário no percentil 85 de valor apostado entre os clientes da operadora seria comparado a quem ocupava o percentil 85 na bwin. A régua passa a ser relativa a cada base, e o modelo volta a operar sobre a informação que efetivamente aprendeu.

Essa é a mesma lógica adotada na construção do escore de intensidade durante o desenvolvimento, quando quatro variáveis de escalas incompatíveis precisaram ser combinadas.

A mitigação reduz o problema, mas não o elimina. Ela pressupõe que o formato da distribuição seja comparável entre as duas populações — suposição razoável, porém não verificada.

---

## Dois modos de aplicação

### Como referência

Utilizar o modelo tal como distribuído, com normalização por percentil sobre a base local.

Adequado para exploração inicial, prova de conceito ou comparação metodológica. O desempenho real será inferior ao reportado, em medida que depende de quão distinta seja a população.

### Como ponto de partida para retreinamento

Utilizar o pipeline, as definições de variáveis e a metodologia, alimentando com dados próprios e reexecutando o ajuste de hiperparâmetros.

É o modo recomendado para qualquer aplicação operacional. Permite ainda incorporar variáveis ausentes na base bwin — duração de sessão, depósitos por sessão e esgotamento de saldo são indicadores apontados pela literatura que a base original não registra, e cuja inclusão tenderia a melhorar o desempenho.

A contribuição principal deste trabalho está na metodologia, não no arquivo do modelo. O classificador serializado é a instância de referência que demonstra a viabilidade da abordagem.

---

## Comportamento em situações específicas

### Valores fora da faixa de treino

O modelo é baseado em árvores de decisão e não extrapola linearmente. Entradas acima do máximo observado no treinamento saturam na predição da folha mais extrema.

A predição permanece válida, mas não diferencia graus além do limite conhecido: um usuário com movimentação dez vezes superior ao máximo da base recebe a mesma probabilidade de quem está no topo da faixa conhecida. A função de predição sinaliza esses casos em sua lista de avisos.

### Valores ausentes

O pipeline substitui ausências pela mediana do treino, sem emitir aviso próprio. Uma predição com três das doze variáveis ausentes retorna com aparência de confiabilidade idêntica a uma predição completa.

A função `prever_risco` incluída no projeto verifica a presença das doze variáveis e interrompe a execução com erro explícito quando alguma falta. Implementações que acessem o pipeline diretamente devem replicar essa verificação.

### Usuários com histórico mínimo

Três variáveis são estatisticamente indefinidas para quem tem um único dia ativo — desvio padrão, tendência de crescimento e concentração. Na base de treino, esses casos foram preenchidos com zero, o que corresponde a 6% dos usuários.

Classificações de usuários muito recentes carregam, portanto, menos informação que o modelo espera. Convém tratar esses casos com cautela adicional.

---

## Limitações reconhecidas

### Desempenho inferior na classe de médio risco

O recall dessa classe é de 0,609, contra 0,959 no baixo risco e 0,828 no alto. É também a classe de maior instabilidade entre divisões da validação cruzada.

A causa é metodológica. Enquanto o alto risco corresponde a um evento concreto — a marcação pela operadora —, a fronteira entre baixo e médio resulta de um corte estatístico em distribuição contínua. Usuários imediatamente acima e abaixo do limiar são comportamentalmente quase indistinguíveis.

A análise de interpretabilidade confirmou essa natureza: a relação entre as variáveis preditoras e a classe de médio risco é não monotônica, com valores extremos em ambas as direções afastando da classe e apenas a região intermediária aproximando.

### Erro assimétrico por decisão de projeto

Quando erra, o modelo superestima a severidade. Entre os casos reais de médio risco, 37,7% são classificados como alto risco, enquanto o erro inverso é raro.

Técnicas de balanceamento foram testadas e reduziram esse desvio em cerca de treze pontos percentuais, mas ao custo de onze a treze pontos no recall da classe de alto risco. O modelo foi mantido sem balanceamento por decisão fundamentada: no contexto de jogo responsável, aplicar intervenção mais restritiva que o necessário causa transtorno, enquanto aplicar intervenção insuficiente compromete a proteção.

Quem adotar o modelo em outro contexto deve avaliar se essa assimetria permanece adequada. Se o custo de falsos positivos for elevado — por exemplo, se a intervenção implicar suspensão prolongada —, a ponderação de classes pode ser preferível.

### Ausência de variáveis relevantes

A base bwin não registra duração de sessão, depósitos por sessão nem esgotamento de saldo. São indicadores apontados pela literatura como relevantes para identificação de comportamento compulsivo, ausentes do modelo por indisponibilidade na fonte.

A tentativa de recuperação de perdas, um dos marcadores mais citados, está representada apenas de forma indireta pelo resultado líquido acumulado e pela taxa de perda — indicadores que não distinguem quem perdeu de quem perseguiu perdas.

### Variável de baixa estabilidade

`tendencia_crescimento_apostado` apresenta distribuição concentrada próxima de zero com caudas que alcançam milhares em ambas as direções. Para usuários com poucos dias ativos muito espaçados, a inclinação da regressão é altamente sensível a observações isoladas.

É a variável de menor importância no modelo final, e essa instabilidade explica parte do resultado.

### Base histórica

Os dados têm quase duas décadas. Padrões de comportamento, produtos ofertados e dinâmica de uso mudaram substancialmente desde então. A validade dos padrões aprendidos em contextos contemporâneos não foi verificada.

---

## Considerações éticas

### A classificação não é diagnóstico

Um usuário classificado como alto risco apresenta padrão comportamental compatível com risco. Não há afirmação sobre condição clínica, e a comunicação com o usuário deve refletir essa distinção.

### Intervenções exigem proporcionalidade e transparência

Medidas restritivas aplicadas automaticamente devem ser proporcionais ao risco identificado e comunicadas de forma clara, com possibilidade de contestação.

Suspensões injustificadas têm efeito contrário ao pretendido: o usuário tende a migrar para plataformas sem qualquer mecanismo de proteção, ampliando a exposição que o sistema buscava reduzir.

### Auditabilidade

O modelo admite análise de interpretabilidade por SHAP, que permite identificar a contribuição de cada variável em predições individuais. Recomenda-se manter registro dessas explicações para decisões que resultem em restrição de acesso.

### Proteção de dados

Dados comportamentais de apostas são sensíveis. Qualquer aplicação sobre usuários identificáveis deve observar a legislação aplicável — no Brasil, a Lei Geral de Proteção de Dados — com atenção à base legal do tratamento, à minimização dos dados coletados e aos direitos do titular.

### Viés detectado e removido durante o desenvolvimento

Uma versão inicial do modelo incluía variáveis demográficas e apresentava desempenho superior. A análise de interpretabilidade revelou que país e idioma dominavam a predição de alto risco — o modelo havia aprendido onde a operadora fiscalizava com mais rigor, não quem apresentava comportamento problemático.

As variáveis demográficas foram removidas, ao custo de dois a quatro pontos percentuais de desempenho. Qualquer retreinamento deve verificar a reincidência desse padrão, que se manifesta como desempenho elevado sustentado por variáveis sem relação causal com o fenômeno.

---

## Requisitos técnicos

Python 3.12 ou superior, com as dependências listadas em `requirements.txt`. O modelo serializado e seus metadados devem permanecer na pasta `modelo/`, em posição relativa ao script que os carrega.

O acesso recomendado é pela função `prever_risco` do módulo `predicao.py`, que implementa as verificações descritas neste documento. O carregamento direto do arquivo serializado contorna essas verificações.

---

## Referências

GRAY, Heather M.; LAPLANTE, Debi A.; SHAFFER, Howard J. Behavioral characteristics of Internet gamblers who trigger corporate responsible gambling interventions. **Psychology of Addictive Behaviors**, v. 26, n. 3, p. 527-535, 2012.

Dados disponíveis publicamente pelo Transparency Project, Division on Addiction, Cambridge Health Alliance — hospital de ensino da Harvard Medical School.

---

## Documentos relacionados

`dicionario-variaveis.md` — especificação das doze variáveis de entrada, com definição operacional, regra de cálculo e faixas observadas.

`relatorio-experimentos.html` — registro cronológico das quatorze etapas de desenvolvimento, com resultados e justificativa de cada decisão.
