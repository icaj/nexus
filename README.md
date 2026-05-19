# ESG Nexus
Plataforma para análise, monitoramento e ranking de aderência às práticas ESG de fornecedores

# Instruções para instalação
1. baixar os arquivos
2. instalar o python
3. digitar o comando 'pip install -r requirements.txt'


# DOCUMENTAÇÃO TÉCNICA — esg_pipeline.py
Pipeline de Análise ESG de Fornecedores
Edenred Brasil | CESAR School 2025


## 1. VISÃO GERAL

O script esg_pipeline.py implementa um pipeline completo de análise ESG
(Environmental, Social and Governance) para classificação de empresas por nível
de maturidade em sustentabilidade.

O pipeline recebe como entrada a base de dados data.csv, contendo 722 empresas
reais já avaliadas e rotuladas pela ESG Enterprise. A partir dessa base, executa
7 etapas sequenciais: pré-processamento dos dados, cálculo das métricas de
maturidade, risco e impacto, análise exploratória, preparação das features,
treino de dois modelos de machine learning (KNN e Random Forest), avaliação dos
modelos e salvamento dos artefatos para uso posterior.

Ao final, os modelos treinados são salvos em disco e utilizados pelo script
esg_predicao.py para classificar novas empresas que entrem no sistema.


## 2. BASE DE DADOS E METODOLOGIA DE ORIGEM

A base de dados utilizada foi rotulada pela ESG Enterprise seguindo a
metodologia descrita no documento ESG-Enterprise-Risk-Ratings-MethodologyV3.pdf.
Esta seção resume os aspectos dessa metodologia que são diretamente relevantes
para o pipeline.

## 2.1 ESTRUTURA DOS SCORES

Cada empresa na base recebe quatro scores numéricos:

  - environment_score  : score do pilar Ambiental     (escala 0 a 1.000)
  - social_score       : score do pilar Social         (escala 0 a 1.000)
  - governance_score   : score do pilar de Governança  (escala 0 a 1.000)
  - total_score        : soma dos três pilares         (escala 0 a 3.000)

A relação entre eles é exata e verificável:

  total_score = environment_score + social_score + governance_score

O script verifica essa condição no pré-processamento e interrompe a execução
caso haja qualquer inconsistência nos dados.

## 2.2 GRADES POR PILAR (escala individual 0–1.000)

Cada pilar recebe uma grade de crédito com base no seu score individual:

  Score         Grade   Nível
  ──────────    ──────  ──────────
  200 a 299     B       Low
  300 a 399     BB      Low
  400 a 499     BBB     High
  500 a 599     A       High
  600 a 899     AA      Excellent
  900 a 1.000   AAA     Excellent

## 2.3 GRADES DO SCORE TOTAL (escala 0–3.000)

O score total (soma dos três pilares) recebe uma grade consolidada:

  Score total   Grade   Nível
  ───────────   ──────  ──────────
  600 a 749     B       Medium
  750 a 899     BB      Medium
  900 a 1.199   BBB     High
  1.200 a 1.799 A       High
  1.800 a 2.699 AA      Excellent
  2.700 a 3.000 AAA     Excellent

Nota: a base de dados disponível contém apenas empresas com grades B, BB, BBB
e A no score total (scores entre 600 e 1.536). Não há empresas com grades AA
ou AAA na base.

## 2.4 PROCESSO DE CÁLCULO DA ESG ENTERPRISE (5 ETAPAS)

O PDF descreve cinco etapas metodológicas para chegar ao score final:

  Etapa 1 — ESG Industry Scores
    Coleta de dados por grupo de indústria a partir de mais de 100 fontes
    externas (relatórios de ESG, NGOs, bases governamentais, mídia social).
    Os dados brutos são normalizados em percentis de 0% a 100% dentro de
    cada indústria. O score da indústria é calculado como a média dos valores
    de todas as empresas do grupo.

  Etapa 2 — Materiality Factors (Matriz de Materialidade)
    Cada KPI recebe um peso de materialidade específico para a indústria,
    variando de 1 a 1.000. Os pesos refletem o quanto aquele indicador é
    relevante para o setor. As categorias ambientais e sociais têm pesos
    ajustados por país e indústria. A categoria de Governança é aplicada
    uniformemente a todos os países.

  Etapa 3 — ESG Materiality Matrix Scores
    Os scores dos pilares E, S e G são calculados aplicando os pesos de
    materialidade sobre os KPI scores. A fórmula é:
    Score do pilar = (score da categoria / soma dos scores) × peso da categoria

  Etapa 4 — Controversies Scores
    Empresas envolvidas em controvérsias ESG recebem penalização no score.
    O score de controvérsia é calculado com base em 50 tópicos e ajustado
    pelo tamanho da empresa (empresas menores recebem ajuste maior, pois
    têm menos cobertura de mídia convencional).

  Etapa 5 — ESG Score Final
    O score final é a média entre o ESG Score calculado e o score de
    controvérsias quando houver controvérsias no período. Quando não há
    controvérsias, o ESG Score permanece igual ao calculado nas etapas
    anteriores.


## 3. ETAPA 1 — PRÉ-PROCESSAMENTO

## 3.1 REMOÇÃO DE COLUNAS NÃO ANALÍTICAS

As colunas de metadados operacionais que não contribuem para a análise são
removidas: ticker, logo, weburl, last_processing_date, cik, currency, exchange.
Após a remoção, a base passa de 21 para 14 colunas.

## 3.2 TRATAMENTO DE NULOS EM INDUSTRY

Treze registros não possuem o campo industry preenchido. Esses registros
recebem o valor "Unknown". Isso evita que esses registros sejam descartados
e permite que ainda participem do treino e da análise, recebendo os pesos
globais no cálculo de risco e impacto.

## 3.3 NORMALIZAÇÃO DE NOMES DE INDÚSTRIA

A base contém variações textuais para o mesmo setor, causadas por uso
inconsistente de "&" versus "and", presença de vírgulas e espaços extras.
O pipeline trata dois tipos de problema:

  Problema 1 — Espaços extras
    A instrução str.strip() remove espaços no início e no final de todos os
    valores da coluna industry. Isso resolve, por exemplo, o caso de "Energy"
    (com espaço ao final) que aparecia como um setor separado de "Energy".
    Após o strip, os 2 registros com espaço são fundidos com os 19 do grupo
    principal, resultando em 21 registros no setor Energy.

  Problema 2 — Variações de nomenclatura
    Um dicionário de nomes canônicos mapeia variações para uma forma padrão.
    A regra adotada é usar "and" no lugar de "&" e remover vírgulas.
    Os três pares tratados são:

      'Aerospace & Defense'            →  'Aerospace and Defense'
      'Hotels, Restaurants & Leisure'  →  'Hotels Restaurants and Leisure'
      'Metals & Mining'                →  'Metals and Mining'

    Resultado: a base passa de 47 setores únicos para 44 setores únicos.

## 3.4 VERIFICAÇÃO DE INTEGRIDADE

O script verifica que total_score é exatamente igual à soma dos três pilares
para todas as 722 linhas. Se houver qualquer divergência, a execução é
interrompida com mensagem de erro. Essa verificação garante que os cálculos
subsequentes de score ponderado e risco sejam matematicamente consistentes
com a metodologia da ESG Enterprise.


## 4. ETAPA 2 — CÁLCULO DAS MÉTRICAS: MATURIDADE, RISCO E IMPACTO

## 4.1 MATURIDADE

A maturidade de cada empresa NÃO é calculada pelo pipeline. Ela já existe na
base de dados como o campo total_level, produzido pela metodologia auditada da
ESG Enterprise. O pipeline apenas mapeia os valores originais para os rótulos
do projeto:

  total_level original   Rótulo no projeto
  ────────────────────   ─────────────────
  High                   Avançado
  Medium                 Iniciante

Este mapeamento foi adotado porque a base disponível contém apenas dois níveis
de maturidade (High e Medium). A maturidade Avançado corresponde às grades BBB
e A (total_score >= 900) e a maturidade Iniciante corresponde às grades B e BB
(total_score entre 600 e 899).

O campo maturidade é a variável-alvo (y) dos modelos de machine learning.
Por vir de uma fonte externa e auditada, o rótulo é independente dos cálculos
internos do pipeline, o que evita a circularidade de treinar um modelo com
rótulos derivados das mesmas features usadas no treino.

Distribuição na base:
  Avançado  (High)   : 451 empresas — 62,5%
  Iniciante (Medium) : 271 empresas — 37,5%

## 4.2 PESOS POR INDÚSTRIA

Antes de calcular o risco e o impacto, o pipeline calcula o peso relativo de
cada pilar (E, S, G) para cada setor da base. Esses pesos expressam o quanto
cada pilar contribui para explicar o desempenho ESG total dentro daquele setor.

4.2.1 MOTIVAÇÃO

Tratar os três pilares com pesos iguais ignoraria diferenças setoriais
importantes. Uma empresa de Utilities tem exposição ambiental muito maior do
que uma empresa de Financial Services. Um banco tem a Governança como pilar
mais determinante. Usar pesos iguais para todos os setores produziria métricas
de risco e impacto distorcidas.

A própria metodologia da ESG Enterprise usa pesos variáveis por indústria
(Materiality Matrix, Etapa 2 do PDF). Os pesos do pipeline seguem essa mesma
lógica, calculados empiricamente a partir da base de treino.

4.2.2 MÉTODO DE CÁLCULO

Para cada setor com 5 ou mais empresas, o pipeline executa os seguintes passos:

  Passo 1 — Calcular a correlação de Pearson de cada pilar com o total_score
            dentro do grupo de empresas daquele setor.
            A correlação mede o quanto as variações de cada pilar estão
            associadas às variações do score total. Quanto maior a correlação
            de um pilar, mais ele explica o desempenho total no setor.

  Passo 2 — Clipar correlações negativas em zero.
            Correlações negativas não fazem sentido como peso de importância
            (não seria coerente dizer que um pilar "prejudica" o score total).

  Passo 3 — Normalizar as três correlações para que somem 1.
            Isso transforma as correlações em pesos proporcionais.

  Exemplo para o setor Banking (n=29):
    correlação E com total  = 0,89
    correlação S com total  = 0,67
    correlação G com total  = 0,82
    soma das correlações    = 2,38

    w_E = 0,89 / 2,38 = 0,374   (37,4%)
    w_S = 0,67 / 2,38 = 0,280   (28,0%)
    w_G = 0,82 / 2,38 = 0,346   (34,6%)
    soma dos pesos             = 1,000  ✓

4.2.3 REGRA DE FALLBACK

Para setores com menos de 5 empresas, os pesos calculados seriam instáveis por
se basearem em pouquíssimas observações. Nesses casos, o pipeline usa os pesos
globais, calculados sobre toda a base de 722 empresas:

  Pesos globais (fallback):
    w_E = 0,3865  (38,65%)
    w_S = 0,3267  (32,67%)
    w_G = 0,2869  (28,69%)

O mesmo fallback é aplicado ao setor "Unknown" (registros sem indústria) e a
qualquer setor que não seja encontrado na base de treino ao classificar uma
empresa nova.

Na base de 44 setores únicos:
  Setores com pesos empíricos : 30 (n >= 5)
  Setores com fallback        : 14 (n < 5)

4.2.4 EXEMPLOS DE PESOS POR SETOR

Os pesos refletem a realidade setorial de forma coerente:

  Setor                    w_E    w_S    w_G    Interpretação
  ─────────────────────    ─────  ─────  ─────  ───────────────────────────────
  Utilities                0,455  0,292  0,253  Alta exposição ambiental
  Airlines                 0,430  0,248  0,322  E domina; G segundo
  Insurance                0,444  0,351  0,205  E domina; G menos relevante
  Banking                  0,374  0,280  0,346  E e G equilibrados; G relevante
  Financial Services       0,348  0,325  0,327  Os três pilares bem equilibrados
  Logística e Transporte   0,509  0,447  0,043  E e S dominam; G quase nulo
  Professional Services    0,471  0,331  0,198  E forte; G pequeno

4.3 SCORE PONDERADO
────────────────────

Com os pesos por setor calculados, o pipeline calcula o score ponderado de cada
empresa. Esse score é a média ponderada dos três pilares, onde os pesos são os
específicos do setor daquela empresa.

Fórmula:

  score_ponderado = (w_E × environment_score)
                  + (w_S × social_score)
                  + (w_G × governance_score)

O resultado está na escala 0–1.000, que é a escala de cada pilar individual.
Não é a soma dos pilares (que estaria na escala 0–3.000), mas a média ponderada,
mantendo a escala de referência de 1.000 como máximo teórico.

Exemplo (empresa do setor Banking):
  environment_score = 400,  w_E = 0,374
  social_score      = 300,  w_S = 0,280
  governance_score  = 350,  w_G = 0,346

  score_ponderado = (0,374 × 400) + (0,280 × 300) + (0,346 × 350)
                  = 149,6 + 84,0 + 121,1
                  = 354,7

## 4.4 RISCO ESG

O risco mede o gap de não-conformidade ESG da empresa em relação ao máximo
possível da escala. Quanto menor o score ponderado, maior o risco.

Fórmula:

  Risco = (1.000 − score_ponderado) / 1.000 × 100

O denominador 1.000 é o máximo teórico do score ponderado (todos os pilares
no máximo, com qualquer combinação de pesos que some 1).

O resultado é expresso em percentual (0% a 100%):
  Risco = 0%   → empresa com score ponderado máximo (conformidade total)
  Risco = 100% → empresa com score ponderado zero (nenhuma conformidade)

Continuando o exemplo anterior (Banking, score_ponderado = 354,7):

  Risco = (1.000 − 354,7) / 1.000 × 100
        = 645,3 / 1.000 × 100
        = 64,5%

Interpretação: esta empresa ainda tem 64,5% de gap em relação ao máximo
possível de conformidade ESG, ponderado pelo perfil de exposição do seu setor.

Referência de valores na base:
  Grade A   (Avançado) : risco médio ≈ 52%  (melhor desempenho)
  Grade BBB (Avançado) : risco médio ≈ 62%
  Grade BB  (Iniciante): risco médio ≈ 68%
  Grade B   (Iniciante): risco médio ≈ 72%  (pior desempenho)

## 4.5 IMPACTO

O impacto mede a posição relativa da empresa dentro do seu setor, com base
no score ponderado. Expressa o quanto aquela empresa está exposta em relação
às demais empresas do mesmo setor.

Fórmula:

  Impacto = percentil do score_ponderado dentro da indústria × 100

O cálculo do percentil é feito pela função rank(pct=True) do pandas, que
atribui a cada empresa sua posição relativa dentro do grupo da indústria.
O resultado é multiplicado por 100 para ficar na escala 0–100.

Interpretação:
  Impacto = 80 → a empresa está acima de 80% das empresas do seu setor
  Impacto = 20 → a empresa está acima de apenas 20% das empresas do setor
  Impacto = 50 → a empresa está exatamente na mediana do setor

Por que usar o score ponderado (e não apenas o pilar Ambiental como antes)?
Ao usar o score ponderado, empresas de setores com alta relevância de G ou S
também são corretamente posicionadas. Um banco com Governança excelente mas
Ambiental mediano será corretamente avaliado como de alto impacto no seu setor,
pois os pesos do Banking valorizam o pilar G.


## 5. QUADRANTE — MATRIZ DE CRITICIDADE

## 5.1 CONCEITO

A Matriz de Criticidade é uma ferramenta de priorização que cruza duas
dimensões para cada empresa:

  - Impacto  : posição relativa no setor (escala 0–100)
  - Risco    : gap de não-conformidade ESG (escala 0–100)

O objetivo é identificar em qual grupo de atenção a empresa se enquadra,
orientando a prioridade de ação da Edenred na gestão da cadeia de fornecedores.

## 5.2 DEFINIÇÃO DOS QUADRANTES

O eixo de corte para ambas as dimensões é 50, que representa a mediana.
Valores acima de 50 são considerados "altos" e abaixo de 50, "baixos".

  Quadrante                        Impacto   Risco    Ação recomendada
  ─────────────────────────────    ────────  ───────  ─────────────────────────
  Alto Impacto / Alto Risco        > 50      > 50     Ação imediata e prioritária
  Alto Impacto / Baixo Risco       > 50      <= 50    Engajamento e manutenção
  Baixo Impacto / Alto Risco       <= 50     > 50     Monitoramento e capacitação
  Baixo Impacto / Baixo Risco      <= 50     <= 50    Monitoramento leve

## 5.3 LÓGICA DE CLASSIFICAÇÃO

A classificação é feita por uma função que recebe os valores de impacto e risco
de cada empresa e aplica as quatro condições em ordem:

  Se impacto > 50 E risco > 50   → "Alto Impacto / Alto Risco"
  Se impacto > 50 E risco <= 50  → "Alto Impacto / Baixo Risco"
  Se impacto <= 50 E risco > 50  → "Baixo Impacto / Alto Risco"
  Caso contrário                 → "Baixo Impacto / Baixo Risco"

## 5.4 EXEMPLO PRÁTICO

Usando os valores calculados no exemplo anterior (Banking):
  score_ponderado = 354,7
  risco           = 64,5%   (> 50 → alto risco)
  impacto         = 62,2    (> 50 → alto impacto, empresa acima da mediana)

  Quadrante resultante: Alto Impacto / Alto Risco → Ação Imediata

## 5.5 DISTRIBUIÇÃO NA BASE DE 722 EMPRESAS

  Quadrante                        Empresas
  ─────────────────────────────    ────────
  Alto Impacto / Alto Risco            ~180
  Alto Impacto / Baixo Risco           ~180
  Baixo Impacto / Alto Risco           ~180
  Baixo Impacto / Baixo Risco          ~182

A distribuição é aproximadamente equilibrada entre os quatro quadrantes porque
tanto o risco quanto o impacto são calculados em relação à própria base —
o impacto usa percentis intra-setor e o risco usa uma fórmula linear sobre
o score ponderado, que por sua vez tem distribuição aproximadamente simétrica.


## 6. ETAPA 3 — ANÁLISE EXPLORATÓRIA

Gera o arquivo analise_exploratoria.png com sete painéis:

  Painel 1 — Distribuição de maturidade (donut chart)
    Proporção de empresas Avançado vs Iniciante na base.

  Painel 2 — Empresas por grade total (barras)
    Contagem de empresas por grade B, BB, BBB, A.

  Painel 3 — Distribuição dos scores por pilar (histograma sobreposto)
    Histogramas dos três pilares para visualizar amplitude e sobreposição.

  Painel 4 — Environment Score × Social Score colorido por maturidade
    Scatter plot que revela a separação visual entre High e Medium.

  Painel 5 — Score ESG médio por indústria (top 12, barras horizontais)
    Ranking dos setores com maior e menor desempenho ESG médio.

  Painel 6 — Matriz de Criticidade (scatter colorido por quadrante)
    Visualização dos 722 pontos distribuídos nos quatro quadrantes,
    com linha de corte em 50 nos dois eixos.

  Painel 7 — Distribuição do risco por grade (boxplot)
    Variação do risco dentro de cada grade, confirmando que o risco
    diminui conforme a grade melhora.


## 7. ETAPA 4 — PREPARAÇÃO DAS FEATURES PARA MACHINE LEARNING

## 7.1 FEATURES DE ENTRADA (X)

Os modelos de machine learning recebem quatro features:

  environment_score  (numérica, 0–1.000)
  social_score       (numérica, 0–1.000)
  governance_score   (numérica, 0–1.000)
  industry_enc       (numérica, resultado do encoding da indústria)

O total_score NÃO é incluído como feature porque total_score = E + S + G.
Incluir os três pilares e o total junto criaria multicolinearidade perfeita —
o modelo aprenderia uma identidade matemática em vez de padrões.

A coluna industry é transformada por LabelEncoder em um número inteiro
(encoding ordinal), tornando-a compatível com os modelos.

## 7.2 VARIÁVEL-ALVO (y)

A variável-alvo é total_level, codificada por LabelEncoder:
  Medium = 0
  High   = 1

## 7.3 DIVISÃO TREINO/TESTE

A base é dividida em 80% para treino (577 empresas) e 20% para teste
(145 empresas). A divisão é estratificada, o que garante que a proporção
entre High e Medium seja mantida em ambos os conjuntos.


## 8. ETAPA 5 — TREINO DOS MODELOS

8.1 KNN (K-NEAREST NEIGHBORS)
───────────────────────────────

O KNN classifica uma empresa nova calculando a distância euclidiana entre seu
vetor de features e todos os vetores do conjunto de treino. Em seguida, vota
entre os k vizinhos mais próximos e atribui a classe mais frequente.

Pré-processamento necessário para o KNN:
  As features têm escalas muito diferentes (scores 0–1.000, industry_enc 0–43).
  Sem normalização, os scores dominariam completamente o cálculo de distância
  e a feature de indústria teria efeito quase nulo. Por isso, o pipeline aplica
  StandardScaler antes do KNN, transformando cada feature para média zero e
  desvio padrão um.

Busca do melhor k:
  O pipeline testa valores de k de 3 a 21 (ímpares) usando validação cruzada
  estratificada de 5 folds. O k com maior acurácia média é selecionado.

Resultado obtido na base:
  Melhor k    : 13
  Acurácia CV : 95,2%
  Acurácia teste: 95,2%

## 8.2 RANDOM FOREST

O Random Forest treina múltiplas árvores de decisão em subconjuntos aleatórios
dos dados (técnica de bagging) e combina as predições por votação majoritária.
É mais robusto que uma única árvore de decisão, pois reduz o overfitting pela
diversidade entre as árvores.

Vantagens em relação ao KNN neste contexto:
  1. Não é afetado pela diferença de escala entre features (não precisa de
     StandardScaler).
  2. Produz importância de variáveis (Gini importance), o que permite saber
     quais pilares são mais determinantes para a classificação.
  3. Lida melhor com a alta correlação entre as features (os três pilares
     têm correlação alta entre si e com o total_score).

Busca de hiperparâmetros (Grid Search com 5-fold estratificado):
  n_estimators      : 100 ou 200 árvores
  max_depth         : 4, 6, 8 ou sem limite
  min_samples_leaf  : 5, 10 ou 15 amostras por folha

Resultado obtido na base:
  Acurácia CV   : 97,9%
  Acurácia teste: 97,9%

Importância das variáveis (valores aproximados):
  environment_score : 64,6%  (pilar mais determinante)
  social_score      : 27,4%
  governance_score  :  7,3%
  industry_enc      :  0,7%


## 9. ETAPA 6 — AVALIAÇÃO DOS MODELOS

A avaliação gera o arquivo avaliacao_modelos.png com três painéis:

  Painel 1 — Matriz de confusão do KNN
    Mostra quantas empresas foram corretamente classificadas como High e Medium,
    e quantas foram confundidas entre as duas classes.

  Painel 2 — Matriz de confusão do Random Forest
    Idem para o Random Forest. Geralmente apresenta menos erros que o KNN.

  Painel 3 — Importância das variáveis (Random Forest)
    Gráfico de barras horizontais com a importância de cada feature,
    confirmando o domínio do pilar Ambiental.

O comparativo final impresso no terminal resume as acurácias de ambos os modelos
no conjunto de teste, permitindo decidir qual usar na predição.


## 10. ETAPA 7 — SALVAMENTO DOS ARTEFATOS

Todos os artefatos são salvos na pasta saida_esg/:

  modelo_knn.pkl
    Contém o modelo KNN treinado, o LabelEncoder da variável-alvo, o
    LabelEncoder da indústria, a lista de features e toda a base de referência
    (para benchmarking por vizinhança na predição).

  modelo_rf.pkl
    Contém o modelo Random Forest treinado, os encoders, a lista de features,
    as importâncias das variáveis e a base de referência.

  config.pkl
    Contém os encoders, a lista de features, o benchmark médio por indústria,
    o mapa de maturidade, os pesos por indústria (calculados na Etapa 2) e o
    threshold mínimo de empresas por setor.

  pesos_por_industria.csv
    Tabela com os pesos w_E, w_S, w_G de cada setor, o número de empresas
    usado no cálculo e a indicação de pesos empíricos vs fallback. Serve como
    documentação metodológica dos pesos adotados.

  base_processada.csv
    A base completa com todas as colunas originais mais as métricas calculadas:
    score_ponderado, maturidade, risco, impacto e quadrante.

  analise_exploratoria.png
    Painel gráfico com sete visualizações da base (Etapa 3).

  avaliacao_modelos.png
    Matrizes de confusão e importância de variáveis (Etapa 6).


## 11. COMO EXECUTAR

Pré-requisitos:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib

Estrutura de arquivos necessária:
  esg_pipeline.py       (este script)
  esg_predicao.py       (script de predição para novas empresas)
  data.csv              (base de 722 empresas)

Execução do pipeline (treino):
  python esg_pipeline.py

Execução da predição (novas empresas):
  python esg_predicao.py --arquivo nova_empresa.csv
  python esg_predicao.py --interativo

O arquivo de entrada para predição deve conter as colunas:
  name, industry, environment_score, social_score, governance_score


## 12. LIMITAÇÕES E CONSIDERAÇÕES METODOLÓGICAS

1. RÓTULO BINÁRIO
   A base disponível tem apenas dois níveis de maturidade (High e Medium).
   Um sistema com mais granularidade (ex: três ou cinco níveis) exigiria
   uma base rotulada com esses níveis adicionais.

2. AUTODECLARAÇÃO
   O script esg_predicao.py aceita scores informados pelo próprio fornecedor.
   Esses scores são autodeclarados e não verificados por auditoria externa,
   ao contrário dos scores da base de treino. Isso deve ser declarado como
   limitação ao apresentar os resultados de predição.

3. PESOS POR INDÚSTRIA
   Os pesos são calculados a partir da correlação linear intra-setor. Para
   setores com poucos representantes (n < 5), os pesos globais são usados como
   fallback. Conforme a base crescer, os pesos empíricos de mais setores se
   tornarão disponíveis.

4. DOMÍNIO DO PILAR AMBIENTAL
   O Random Forest indica que o pilar Ambiental responde por 64,6% do poder
   preditivo de maturidade. Isso pode refletir a estrutura real dos dados ou
   um viés da base. Com mais dados e setores diversos, essa proporção pode
   se distribuir de forma mais equilibrada.

5. RETREINO PERIÓDICO
   Os modelos foram treinados com a base disponível em determinado momento.
   Recomenda-se retreinar periodicamente conforme novos dados de empresas
   forem incorporados à base, para manter a relevância dos padrões aprendidos.


Fim da documentação — esg_pipeline.py
Edenred Brasil | CESAR School 2025
