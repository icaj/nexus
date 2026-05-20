# DOCUMENTAÇÃO TÉCNICA — esg_predicao.py
Script de Classificação de Novos Fornecedores
| Edenred Brasil | CESAR School 2025 |

# 1. VISÃO GERAL

O script esg_predicao.py é a etapa de uso dos modelos treinados pelo
esg_pipeline.py. Enquanto o pipeline processa a base histórica de 722 empresas
e treina os modelos, este script aplica esses modelos sobre empresas novas que
ainda não foram avaliadas.

Para cada nova empresa fornecida, o script executa um pipeline completo de
classificação em sequência: carrega os modelos salvos, recebe os três scores
ESG da empresa (Ambiental, Social e Governança), calcula o score ponderado pelo
setor, calcula o risco e o impacto, classifica o nível de maturidade via KNN e
via Random Forest, identifica as 3 empresas mais similares da base para
benchmarking e gera um plano de ação priorizado pelos pilares mais deficientes.

Pré-requisito obrigatório:
  O script esg_pipeline.py deve ter sido executado antes, pois os modelos
  salvos em saida_esg/ (modelo_knn.pkl, modelo_rf.pkl, config.pkl) são
  carregados no início da execução. Sem esses arquivos, o script não funciona.

Modos de execução:
  python esg_predicao.py --arquivo nova_empresa.csv
  python esg_predicao.py --interativo


# 2. METODOLOGIA DE REFERÊNCIA — ESG ENTERPRISE

Os modelos foram treinados com dados rotulados pela ESG Enterprise seguindo a
metodologia do documento ESG-Enterprise-Risk-Ratings-MethodologyV3.pdf. Este
script aplica a mesma estrutura de scores e grades ao classificar novas
empresas.

## 2.1 ESCALA DE SCORES

Cada empresa é avaliada em três pilares, cada um na escala de 0 a 1.000:

  environment_score  : pilar Ambiental     (0 a 1.000)
  social_score       : pilar Social        (0 a 1.000)
  governance_score   : pilar Governança    (0 a 1.000)

O score total é a soma dos três pilares e varia de 0 a 3.000:
  total_score = environment_score + social_score + governance_score

## 2.2 GRADES DE REFERÊNCIA POR PILAR (escala 0–1.000 cada)

Cada pilar individual é enquadrado em uma grade com base no seu score:

  ! Score ! Grade | Interpretação |
  !-------|-------|---------------|
  | < 200 | <B | Abaixo do mínimo avaliável |
  | 200 a 299 | B | Desempenho baixo |
  | 300 a 399 | BB | Desempenho baixo-médio |
  | 400 a 499 | BBB | Desempenho médio-alto |
  | 500 a 599 | A | Desempenho alto |
  | 600 a 899 | AA | Desempenho excelente |
  | 900 a 1.000 | AAA | Desempenho máximo |

## 2.3 GRADES DO SCORE TOTAL (escala 0–3.000)

O score total também recebe uma grade consolidada:

  | Score total | Grade | Nível | Maturidade no projeto |
  |-------------|-------|-------|-----------------------|
  | 0 a 599 | D | Abaixo mín. | — |
  | 600 a 749 | B | Baixo | Iniciante (Medium) |
  | 750 a 899 | BB | Baixo-Médio | Iniciante (Medium) |
  | 900 a 1.199 | BBB | Médio-Alto | Avançado (High) |
  | 1.200 a 1.799 | A | Alto | Avançado  (High) |
  | 1.800 a 2.699 | AA | Excelente | — |
  | 2.700 a 3.000 | AAA | Máximo | — |

Nota: a base de treino dos modelos contém apenas empresas com total_score
entre 600 e 1.536 (grades B a A). As grades AA e AAA existem na escala
teórica mas não possuem representantes nos modelos treinados.

## 2.4 LIMIAR DE PLANO DE AÇÃO (score por pilar < 400)

O limiar de 400 pontos por pilar corresponde ao início da grade BBB, que é
o ponto de entrada para o nível High na metodologia ESG Enterprise. Pilares
abaixo de 400 estão em BB ou inferior, o que indica necessidade de intervenção.
Por isso, o plano de ação é gerado apenas para pilares com score < 400.


# 3. SCORES — CÁLCULO E APRESENTAÇÃO

## 3.1 ENTRADA DE DADOS

O script recebe três scores por empresa, na escala ESG Enterprise (0–1.000):
  - environment_score : score do pilar Ambiental
  - social_score      : score do pilar Social
  - governance_score  : score do pilar de Governança

Esses valores podem ser fornecidos via arquivo CSV/Excel ou digitados
interativamente no terminal.

## 3.2 SCORE TOTAL

O score total é calculado como a soma simples dos três pilares:

  total_score = environment_score + social_score + governance_score

Esse valor é usado para mapear a grade de referência (B, BB, BBB, A) e para
o cálculo do nível e da maturidade de forma direta.

## 3.3 SCORE PONDERADO

O score ponderado é a média ponderada dos três pilares com pesos específicos
do setor da empresa. É a métrica central usada para calcular risco e impacto.

Fórmula:

  score_ponderado = (w_E × environment_score)
                  + (w_S × social_score)
                  + (w_G × governance_score)

Onde w_E + w_S + w_G = 1 e os pesos variam de acordo com o setor da empresa.

O resultado está na escala 0–1.000 (mesma escala de cada pilar individual),
pois é uma média ponderada — não uma soma.

## 3.4 COMO OS PESOS DO SETOR SÃO OBTIDOS

O script recupera os pesos calculados durante o treino do pipeline (salvos em
config.pkl) para o setor informado. Se o setor existir na tabela de pesos
empíricos, seus pesos específicos são usados. Caso contrário, são usados os
pesos globais como fallback.

Pesos globais (fallback):
  w_E = 0,3865  (Ambiental  — 38,65%)
  w_S = 0,3267  (Social     — 32,67%)
  w_G = 0,2869  (Governança — 28,69%)

Exemplos de pesos empíricos por setor:

  | Setor | w_E | w_S | w_G |
  |-------|-----|-----|-----|
  | Utilities | 0,455 | 0,292 | 0,253 |
  | Airlines | 0,430 | 0,248 | 0,322 |
  | Banking | 0,374 | 0,280 | 0,346 |
  | Financial Services | 0,348 | 0,325 | 0,327 |
  | Technology | 0,391 | 0,318 | 0,291 |

O relatório impresso informa a origem dos pesos utilizados:
  "empírico (n=29)"  → pesos calculados com 29 empresas do setor na base
  "fallback (n=3)"   → setor com menos de 5 empresas; usados pesos globais
  "fallback (setor não encontrado)" → setor não presente na base de treino

## 3.5 GRADE DE REFERÊNCIA POR PILAR

No relatório, cada pilar recebe sua grade de referência individual com base
no score informado, usando a tabela da metodologia ESG Enterprise:

  | Score do pilar | Grade exibida |
  |----------------|---------------|
  | >= 600 | AA |
  | 500 a 599 | A |
  | 400 a 499 | BBB |
  | 300 a 399 | BB |
  | 200 a 299 | B |
  | < 200 | <B |


# 4. MATURIDADE — CLASSIFICAÇÃO VIA KNN E RANDOM FOREST

## 4.1 O QUE É MATURIDADE NO PROJETO

A maturidade representa o nível de desenvolvimento das práticas ESG de uma
empresa. O projeto usa dois níveis, derivados da metodologia ESG Enterprise:

  | Nível ESG Enterprise | Maturidade no projeto |
  |----------------------|-----------------------|
  | High | Avançado |
  | Medium | Iniciante |

Empresas Avançadas (High) têm total_score >= 900 (grade BBB ou superior).
Empresas Iniciantes (Medium) têm total_score entre 600 e 899 (grade B ou BB).

## 4.2 VETOR DE FEATURES PARA OS MODELOS

Ambos os modelos recebem o mesmo vetor de quatro features:

  [environment_score, social_score, governance_score, industry_enc]

onde industry_enc é o código numérico do setor, gerado pelo LabelEncoder
treinado no pipeline e carregado do arquivo config.pkl.

## 4.3 CLASSIFICAÇÃO VIA KNN

O KNN (K-Nearest Neighbors) classifica a empresa nova calculando a distância
euclidiana entre seu vetor de features e todos os 722 vetores da base de
treino. Os k vizinhos com menor distância são selecionados e votam na classe
mais frequente entre eles.

Detalhes técnicos:
  - O melhor k foi determinado por validação cruzada no pipeline (k=13)
  - Antes do cálculo de distância, as features passam por StandardScaler
    para que scores (0–1000) e industry_enc (0–43) fiquem na mesma escala
  - A acurácia obtida na base de teste foi de 95,2%

Saída do KNN:
  - Maturidade predita: High (Avançado) ou Medium (Iniciante)
  - Confiança por classe: percentual de votos entre os k vizinhos
  - 3 vizinhos mais próximos: empresas mais similares para benchmarking

Por que o KNN também serve para benchmarking:
  Os vizinhos retornados não são apenas auxiliares da predição — eles são
  empresas reais da base de treino com perfil de scores mais parecido com
  o da empresa nova. Isso permite dizer "sua empresa tem perfil similar ao
  da Empresa X, Y e Z", criando um benchmarking contextualizado.

## 4.4 CLASSIFICAÇÃO VIA RANDOM FOREST

O Random Forest treina múltiplas árvores de decisão em subconjuntos aleatórios
da base (bagging) e combina as predições por votação majoritária.

Detalhes técnicos:
  - Hiperparâmetros otimizados por GridSearchCV no pipeline
  - A acurácia obtida na base de teste foi de 97,9%
  - Não requer normalização de features (não usa cálculo de distância)

Saída do Random Forest:
  - Maturidade predita: High (Avançado) ou Medium (Iniciante)
  - Confiança por classe: proporção de árvores que votaram em cada classe
  - Importância das variáveis: usada para priorizar o plano de ação

Importância das variáveis (valores obtidos no treino):
  environment_score : 64,6%  — pilar mais determinante para maturidade
  social_score      : 27,4%
  governance_score  :  7,3%
  industry_enc      :  0,7%

Essa importância orienta diretamente a ordem dos planos de ação: o pilar
com maior importância e score abaixo do limiar aparece primeiro.

## 4.5 INTERPRETAÇÃO DOS DOIS MODELOS JUNTOS

Apresentar os dois modelos simultaneamente é uma escolha deliberada. Quando
ambos concordam na classificação, a confiança na predição é maior. Quando
divergem, isso sinaliza que a empresa está em uma região de fronteira — com
scores próximos ao limiar entre Avançado e Iniciante — e merece atenção
especial na avaliação.


# 5. RISCO ESG — CÁLCULO PONDERADO

## 5.1 DEFINIÇÃO DE RISCO NO PROJETO

O risco ESG mede o gap de não-conformidade da empresa em relação ao máximo
teórico da escala. Quanto menor o score ponderado, maior o risco — pois a
empresa está mais distante do estado ideal de conformidade ESG.

Não é risco financeiro. É o risco de a empresa não atender aos requisitos
de uma boa gestão dos pilares Ambiental, Social e de Governança. Essa
distinção está alinhada com a definição de risco ESG não-financeiro do
COSO & WBCSD (2018).

## 5.2 POR QUE USAR O SCORE PONDERADO E NÃO O SCORE TOTAL

O score total (soma dos três pilares, escala 0–3.000) trata os três pilares
como igualmente importantes para qualquer empresa de qualquer setor. Isso
seria impreciso: uma empresa de Utilities tem exposição ambiental muito maior
do que uma empresa de Financial Services.

O score ponderado incorpora os pesos específicos do setor, fazendo com que
o risco calculado reflita o que é mais relevante para aquele tipo de empresa.
Uma empresa de Banking com Governança fraca terá risco maior do que uma
empresa de Airlines com a mesma Governança fraca — porque G tem mais peso
no setor bancário do que no setor aéreo.

## 5.3 FÓRMULA DO RISCO

  Risco = (1.000 − score_ponderado) / 1.000 × 100

O denominador 1.000 é o máximo teórico do score ponderado (quando todos os
pilares atingem o máximo de 1.000 com qualquer combinação de pesos que some 1).
O resultado é expresso em percentual (0% a 100%).

Interpretação:
  Risco = 0%   → score ponderado = 1.000 (conformidade ESG máxima)
  Risco = 50%  → score ponderado = 500   (metade do potencial ESG)
  Risco = 100% → score ponderado = 0     (nenhuma conformidade ESG)

## 5.4 EXEMPLO COMPLETO DE CÁLCULO

Empresa: "Banco Exemplo S.A."
Setor: Banking
Scores: environment_score=350, social_score=280, governance_score=420

Passo 1 — Obter pesos do setor Banking:
  w_E = 0,374  (Ambiental)
  w_S = 0,280  (Social)
  w_G = 0,346  (Governança)

Passo 2 — Calcular score ponderado:
  score_ponderado = (0,374 × 350) + (0,280 × 280) + (0,346 × 420)
  score_ponderado = 130,9 + 78,4 + 145,3
  score_ponderado = 354,6

Passo 3 — Calcular risco:
  Risco = (1.000 − 354,6) / 1.000 × 100
  Risco = 645,4 / 1.000 × 100
  Risco = 64,5%

Interpretação: o Banco Exemplo S.A. tem 64,5% de gap em relação ao máximo
possível de conformidade ESG, considerando o perfil de exposição do setor
bancário.

## 5.5 FAIXAS DE RISCO DE REFERÊNCIA NA BASE

  | Grade ESG | Risco médio | Interpretação |
  |-----------|-------------|---------------|
  | A | ≈ 52% | Melhor desempenho da base |
  | BBB | ≈ 62% | Nível alto (Avançado) |
  | BB | ≈ 68% | Nível baixo-médio (Iniciante) |
  | B | ≈ 72% | Pior desempenho da base |


# 6. IMPACTO — POSIÇÃO RELATIVA NO SETOR

## 6.1 DEFINIÇÃO DE IMPACTO

O impacto mede onde a empresa está em relação às demais empresas do mesmo
setor, com base no score ponderado. Expressa a exposição relativa da empresa
no contexto setorial.

## 6.2 FÓRMULA

O cálculo usa a média e o desvio padrão do score ponderado no setor como
referência, transformando o score da empresa em uma posição na escala 0–100:

  z = (score_ponderado − média_setor) / desvio_padrão_aproximado
  impacto = 50 + z × 25
  impacto = max(0, min(100, impacto))  [limitado entre 0 e 100]

Os parâmetros utilizados:
  média_setor     : média do score_ponderado das empresas do setor na base
                    de treino (recuperada do benchmark salvo em config.pkl)
  desvio_padrão   : valor aproximado de 110, estimado da base de treino
  valor 50        : representa a mediana (empresa na média do setor)
  fator 25        : escala o desvio para a faixa 0–100

Interpretação:
  Impacto > 50 → empresa está acima da média do seu setor
  Impacto = 50 → empresa está na mediana do seu setor
  Impacto < 50 → empresa está abaixo da média do seu setor

## 6.3 POR QUE USAR O SCORE PONDERADO NO IMPACTO

O uso do score ponderado (em vez de apenas o pilar Ambiental, como em versões
anteriores) garante que o impacto seja coerente com os pesos setoriais. Um
banco com Governança excelente mas Ambiental mediano será corretamente
avaliado como de alto impacto no seu setor, pois o pilar G tem mais peso
no Banking. Usar apenas o pilar E subavaliaria esse banco.


# 7. QUADRANTE — MATRIZ DE CRITICIDADE

## 7.1 CONCEITO

A Matriz de Criticidade é uma ferramenta de priorização que posiciona a empresa
em um dos quatro quadrantes formados pelo cruzamento de Impacto × Risco.
É o principal entregável estratégico do script — define qual ação a Edenred
deve tomar em relação a cada fornecedor avaliado.

## 7.2 EIXOS DA MATRIZ

  Eixo X (horizontal) — Impacto: posição relativa da empresa no setor (0–100)
  Eixo Y (vertical)   — Risco  : gap de não-conformidade ESG (0–100%)

O ponto de corte para ambos os eixos é 50, que representa a mediana.

## 7.3 OS QUATRO QUADRANTES

  | Quadrante | Condição | Ação recomendada |
  |-----------|----------|------------------|
  | Alto Impacto / Alto Risco | impacto>50 e risco>50 | Ação imediata e prioritária |
  | Alto Impacto / Baixo Risco | impacto>50 e risco<=50 | Engajamento e manutenção |
  | Baixo Impacto / Alto Risco | impacto<=50 e risco>50 | Monitoramento e capacitação |
  | Baixo Impacto / Baixo Risco | impacto<=50 e risco<=50 | Monitoramento leve |

## 7.4 LÓGICA DE CLASSIFICAÇÃO

A classificação é feita pela função classificar_quadrante, que recebe os
valores de impacto e risco e aplica as condições em ordem:

  Se impacto > 50 E risco > 50   → "Alto Impacto / Alto Risco"
  Se impacto > 50 E risco <= 50  → "Alto Impacto / Baixo Risco"
  Se impacto <= 50 E risco > 50  → "Baixo Impacto / Alto Risco"
  Caso contrário                 → "Baixo Impacto / Baixo Risco"

## 7.5 EXEMPLO PRÁTICO COMPLETO

Continuando o exemplo do "Banco Exemplo S.A." (Banking):
  score_ponderado = 354,6
  risco           = 64,5%   → risco > 50: ALTO RISCO

Para o impacto, supondo que a média do setor Banking é 370:
  z       = (354,6 − 370) / 110 = −0,14
  impacto = 50 + (−0,14 × 25)  = 50 − 3,5 = 46,5
            → impacto < 50: BAIXO IMPACTO

Quadrante resultante: Baixo Impacto / Alto Risco → Monitoramento e Capacitação

Interpretação: o banco está abaixo da média do setor e com alto gap de
conformidade ESG. Recomenda-se monitoramento periódico e capacitação nos
pilares mais deficientes, sem necessidade de ação imediata.

## 7.6 RELAÇÃO COM OS ENTREGÁVEIS DO PROJETO

O quadrante conecta diretamente ao entregável "Matriz de Criticidade" definido
na apresentação Edenred_ESG_Apresentacao.pptx (slide 15):

  "Classificação dos fornecedores cruzando impacto (ambiental/social) ×
  probabilidade de não conformidade. Permite priorizar ações de engajamento
  e auditoria nos fornecedores de maior risco."


# 8. PLANO DE AÇÃO

## 8.1 CRITÉRIO DE ATIVAÇÃO

O plano de ação é gerado para pilares com score abaixo de 400 pontos. O limiar
de 400 corresponde ao início da grade BBB na metodologia ESG Enterprise, que
marca a entrada no nível High. Pilares em BB ou inferior (< 400) indicam
necessidade de intervenção.

## 8.2 PRIORIZAÇÃO PELO RANDOM FOREST

Os pilares que atendem ao critério (score < 400) são ordenados pela importância
que o Random Forest aprendeu para cada variável durante o treino. O pilar com
maior importância aparece primeiro no plano, pois é o que mais impacta a
classificação de maturidade.

Ordem de importância aprendida (base de treino):
  1º — Ambiental (64,6% de importância)
  2º — Social (27,4%)
  3º — Governança (7,3%)

## 8.3 AÇÕES POR PILAR

Para cada pilar com score < 400, as três primeiras ações da lista são
apresentadas no plano:

  Pilar Ambiental:
    • Implementar processo formal de gestão de impactos ambientais
    • Realizar levantamento e estimativa da pegada de carbono (GHG Protocol)
    • Buscar certificação ISO 14001 ou equivalente

  Pilar Social:
    • Formalizar política de compromisso com trabalho digno e direitos humanos
    • Implementar programa estruturado de diversidade, equidade e inclusão
    • Criar programa de saúde mental e bem-estar para colaboradores

  Pilar Governança:
    • Aprovar e publicar política formal de responsabilidade socioambiental
    • Implantar código de conduta anticorrupção e canal de denúncias
    • Incluir cláusulas ESG nos contratos com fornecedores

## 8.4 CASO SEM PLANO DE AÇÃO

Se todos os pilares tiverem score >= 400 (grade BBB ou superior), nenhuma
ação é gerada. O relatório indica que a empresa está em nível satisfatório
e recomenda manter as práticas e evoluir para certificações externas.


# 9. BENCHMARKING — 3 EMPRESAS MAIS SIMILARES

## 9.1 COMO FUNCIONA

O benchmarking é um subproduto natural do KNN. Após classificar a empresa
nova, o script recupera os 3 vizinhos mais próximos encontrados pelo modelo
durante a predição — ou seja, as 3 empresas da base de treino cujo vetor de
features tem menor distância euclidiana (após normalização pelo StandardScaler)
ao vetor da empresa nova.

## 9.2 O QUE É EXIBIDO

Para cada uma das 3 empresas mais similares, o relatório exibe:
  - Nome da empresa
  - Setor de atuação
  - Nível de maturidade (Avançado ou Iniciante)
  - Score total na escala ESG Enterprise
  - Distância euclidiana (0 = idêntica, quanto menor mais similar)

## 9.3 INTERPRETAÇÃO PRÁTICA

O benchmarking responde à pergunta: "quais empresas já avaliadas têm perfil
ESG mais parecido com o meu?" Se os vizinhos mais próximos são majoritariamente
Avançados, isso reforça a classificação de maturidade atribuída. Se são
Iniciantes, indica que a empresa está em região de menor desempenho relativo.

O gestor pode ainda usar os vizinhos como referência: "empresas similares à
minha já implementaram estas práticas — o que elas fazem que eu ainda não faço?"


# 10. MODOS DE EXECUÇÃO

## 10.1 MODO ARQUIVO (--arquivo ou -a)

Recebe um arquivo CSV ou Excel com uma ou mais empresas. O arquivo deve conter
obrigatoriamente as seguintes colunas:

  name               : nome da empresa
  industry           : setor de atuação (deve corresponder a um setor da base)
  environment_score  : score do pilar Ambiental (inteiro, 0 a 1.000)
  social_score       : score do pilar Social (inteiro, 0 a 1.000)
  governance_score   : score do pilar de Governança (inteiro, 0 a 1.000)

Execução:
  python esg_predicao.py --arquivo nova_empresa.csv

Para múltiplas empresas, cada linha do arquivo gera um relatório completo
e independente.

## 10.2 MODO INTERATIVO (--interativo ou -i)

O script solicita os dados diretamente pelo terminal, um campo por vez:
nome da empresa, setor e os três scores. Útil para avaliações pontuais sem
necessidade de preparar um arquivo.

Execução:
  python esg_predicao.py --interativo


# 11. ESTRUTURA DO RELATÓRIO DE SAÍDA

Para cada empresa avaliada, o relatório impresso no terminal contém seis seções:

  Seção 1 — SCORES
    Tabela com environment_score, social_score, governance_score, o peso do
    setor para cada pilar, a grade de referência individual, o score ponderado
    e o score total com sua grade consolidada. Informa a origem dos pesos.

  Seção 2 — RISCO ESG
    Fórmula, cálculo passo a passo e resultado em percentual.

  Seção 3 — IMPACTO
    Score ponderado comparado à média do setor e resultado em escala 0–100.

  Seção 4 — QUADRANTE
    Classificação na Matriz de Criticidade com a ação recomendada.

  Seção 5 — MATURIDADE (KNN e Random Forest)
    Nível predito por cada modelo com a confiança percentual por classe.

  Seção 6 — BENCHMARKING
    As 3 empresas mais similares com nome, setor, maturidade, score e distância.

  Seção 7 — PLANO DE AÇÃO
    Ações priorizadas para pilares com score < 400, ordenadas por importância
    aprendida pelo Random Forest.


# 12. ARQUIVOS NECESSÁRIOS PARA EXECUÇÃO

O script depende dos seguintes arquivos gerados pelo esg_pipeline.py:

  saida_esg/modelo_knn.pkl
    Modelo KNN treinado com os 577 exemplos de treino (80% da base).
    Contém o modelo, o LabelEncoder da variável-alvo, o LabelEncoder da
    indústria, a lista de features e a base de referência completa para
    o benchmarking por vizinhança.

  saida_esg/modelo_rf.pkl
    Modelo Random Forest treinado. Contém o modelo, os encoders, a lista
    de features, as importâncias das variáveis e a base de referência.

  saida_esg/config.pkl
    Configurações gerais: encoders, features, benchmark médio por indústria,
    mapa de maturidade e tabela de pesos por indústria.

Se qualquer um desses arquivos estiver ausente, o script exibe mensagem de
erro e solicita que o esg_pipeline.py seja executado primeiro.


# 13. LIMITAÇÕES E CONSIDERAÇÕES

1. ESCALA DE ENTRADA
   Os scores informados devem estar na escala ESG Enterprise (0–1.000 por pilar).
   Scores em outras escalas (ex: 0–100) produzirão resultados incorretos.

2. SETORES NÃO MAPEADOS
   Se o setor informado não existir na base de treino, os pesos globais são
   usados como fallback. O relatório indica essa situação explicitamente.
   Recomenda-se verificar a grafia do setor antes de usar o modo interativo.

3. AUTODECLARAÇÃO
   Os scores informados pelo próprio fornecedor não são auditados externamente,
   ao contrário dos scores da base de treino (ESG Enterprise). Os resultados
   devem ser declarados como baseados em scores autodeclarados.

4. DOIS NÍVEIS DE MATURIDADE
   Os modelos foram treinados com apenas dois níveis (High e Medium). Uma
   classificação mais granular exigiria uma base rotulada com mais categorias.

5. RETREINO PERIÓDICO
   Conforme novos dados de empresas forem incorporados à base de treino, é
   recomendável re-executar o esg_pipeline.py para atualizar os modelos e
   os pesos por indústria com os padrões mais recentes.


================================================================================
Fim da documentação — esg_predicao.py
Edenred Brasil | CESAR School 2025
================================================================================
