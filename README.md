# ESG Nexus Enterprise

Plataforma corporativa para **avaliação ESG (Environmental, Social & Governance) de fornecedores** com Machine Learning, dashboards analíticos, rastreamento de experimentos e orquestração de pipelines.

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Módulos e Componentes](#módulos-e-componentes)
  - [Backend — FastAPI](#backend--fastapi)
  - [Frontend — Streamlit](#frontend--streamlit)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Banco de Dados — PostgreSQL / NeonDB](#banco-de-dados--postgresql--neondb)
  - [MLflow — Rastreamento de Experimentos](#mlflow--rastreamento-de-experimentos)
  - [Apache Airflow — Orquestração](#apache-airflow--orquestração)
- [API — Endpoints](#api--endpoints)
- [Perfis de Usuário e Controle de Acesso](#perfis-de-usuário-e-controle-de-acesso)
- [Variáveis de Ambiente](#variáveis-de-ambiente)
- [Instalação](#instalação)
  - [Opção 1 — Docker Compose (recomendado)](#opção-1--docker-compose-recomendado)
  - [Opção 2 — Instalação local (desenvolvimento)](#opção-2--instalação-local-desenvolvimento)
  - [Opção 3 — Máquina Virtual / Servidor Linux](#opção-3--máquina-virtual--servidor-linux)
- [Primeiro Treinamento do Modelo](#primeiro-treinamento-do-modelo)
- [Atualização em Produção](#atualização-em-produção)
- [Solução de Problemas](#solução-de-problemas)
- [Documentação Adicional](#documentação-adicional)

---

## Visão Geral

O ESG Nexus Enterprise avalia o desempenho ESG de fornecedores combinando:

- **Scores de entrada** (Ambiental, Social, Governança na escala 0–1000) fornecidos via formulário, upload de planilha ou API
- **Pesos por setor** calculados automaticamente a partir do dataset Kaggle ESG (ex.: Energia recebe maior peso Ambiental)
- **Dois modelos ML** treinados com GridSearchCV: **K-Nearest Neighbors** e **Random Forest**, que classificam o fornecedor em `High` ou `Medium` maturidade ESG
- **Plano de ação** gerado automaticamente com ações de melhoria por pilar (E/S/G)
- **Rastreamento completo** de experimentos via MLflow
- **Orquestração diária** via Apache Airflow (re-treino às 02:00)

### Tecnologias principais

| Camada | Tecnologia | Versão mínima |
|--------|-----------|---------------|
| API REST | FastAPI + Uvicorn | 0.110 / 0.29 |
| ORM | SQLAlchemy 2.x | 2.0 |
| Banco de dados | PostgreSQL 16 (NeonDB) | — |
| Driver PostgreSQL | psycopg 3 (extensão C) | 3.2 |
| ML | scikit-learn + joblib | 1.4 / 1.4 |
| Experimentos ML | MLflow | 2.12 |
| Orquestração | Apache Airflow | 2.9.3 |
| Frontend | Streamlit + Plotly | 1.36 / 5.22 |
| Autenticação | JWT (python-jose + bcrypt) | — |
| Linguagem | Python | 3.11 |
| Container | Docker + Docker Compose | 24+ / 2.x |

---

## Arquitetura

O projeto segue a **Arquitetura Hexagonal** (Ports & Adapters), garantindo que o núcleo de domínio seja independente de frameworks e bancos de dados.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          ESG Nexus Enterprise                                 │
│                                                                               │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐ │
│  │   INTERFACES    │    │                  DOMÍNIO                         │ │
│  │                 │    │                                                  │ │
│  │  FastAPI (REST) │◄──►│  Entidades: Empresa, ScoreESG, DiagnosticoESG   │ │
│  │  Streamlit (UI) │    │  Serviços:  preprocessamento, feature_eng,       │ │
│  │                 │    │             treinamento, avaliação, predicao      │ │
│  └────────┬────────┘    │  Portas:    IRepositorioModelo (interface)        │ │
│           │             └──────────────────────────────────────────────────┘ │
│           │                                    ▲                             │
│           ▼                                    │                             │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐ │
│  │   APLICAÇÃO     │    │                 ADAPTADORES                      │ │
│  │                 │    │                                                  │ │
│  │ ServicoAval.    │◄──►│  Entrada:  LeitorKaggle, LeitorFornecedores      │ │
│  │ ServicoTrein.   │    │  Saída:    RepositorioModeloJoblib, CsvResultado  │ │
│  └─────────────────┘    └──────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        INFRAESTRUTURA                                  │  │
│  │  PostgreSQL/NeonDB │ SQLAlchemy ORM │ JWT/bcrypt │ MLflow │ structlog  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Fluxo de dados — Classificação de fornecedor

```
[Usuário / Airflow]
        │
        │  POST /classificar  (JSON com scores E/S/G e setor)
        ▼
[FastAPI — rotas_enterprise.py]
        │
        ├─► Converte PT→EN (setor) e normaliza scores
        ├─► ServicoAvaliacao.avaliar(empresa)
        │       ├─► Carrega modelo_knn.joblib + modelo_rf.joblib + config.joblib
        │       ├─► Feature engineering (pesos por setor)
        │       ├─► Predição KNN → maturidade + confiança
        │       ├─► Predição RF  → maturidade + confiança + importância
        │       └─► Gera DiagnosticoESG (score, grade, risco, impacto, quadrante)
        │
        ├─► Persiste AvaliacaoBanco + PlanoAcaoBanco (PostgreSQL/NeonDB)
        └─► Retorna JSON com diagnóstico completo
```

### Serviços Docker

```
┌──────────────────────────────────────────────────────────┐
│                   docker-compose.yml                     │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │     api     │   │   frontend-  │   │    mlflow    │ │
│  │  :8000      │   │   streamlit  │   │   :5000      │ │
│  │  FastAPI    │◄──│   :8501      │   │              │ │
│  └──────┬──────┘   └──────────────┘   └──────────────┘ │
│         │                                               │
│         │  (NeonDB — externo, AWS us-east-1)            │
│         └──────────────────────────────────────────────►│
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │airflow-init │   │  airflow-    │   │  airflow-    │ │
│  │ (run-once)  │   │  webserver   │   │  scheduler   │ │
│  └─────────────┘   │  :8080       │   │              │ │
│                    └──────────────┘   └──────────────┘ │
│                                                          │
│  ┌─────────────┐                                        │
│  │   trainer   │  (profile: training — execução manual) │
│  │ treinar_    │                                        │
│  │ modelo.py   │                                        │
│  └─────────────┘                                        │
└──────────────────────────────────────────────────────────┘
```

> O serviço `trainer` é ativado apenas com `docker compose --profile training run trainer`.  
> O PostgreSQL local do compose está disponível como serviço `postgres` (porta 5432), mas o projeto usa NeonDB por padrão.

---

## Estrutura de Diretórios

```
ESGNexus/
│
├── esg_ml/                          # Pacote principal (Arquitetura Hexagonal)
│   ├── dominio/                     # Núcleo do negócio — sem dependências externas
│   │   ├── entidades/
│   │   │   ├── empresa.py           # Dataclasses: Empresa, ScoreESG, PesosESG
│   │   │   └── diagnostico.py       # Dataclass: DiagnosticoESG (resultado completo)
│   │   ├── portas/
│   │   │   └── portas.py            # Interface IRepositorioModelo (ABC)
│   │   └── servicos/
│   │       ├── preprocessamento.py  # Limpeza e normalização dos dados Kaggle
│   │       ├── feature_engineering.py  # Cálculo de pesos ESG por setor
│   │       ├── treinamento.py       # GridSearchCV: KNN (k=3..21) + RF (24 combinações)
│   │       ├── avaliacao.py         # Métricas, critério de aprovação (acc≥93%, F1≥88%)
│   │       └── predicao.py          # Inferência para novos fornecedores
│   │
│   ├── aplicacao/                   # Casos de uso (orquestração)
│   │   ├── servico_treinamento.py   # CRISP-DM Fases 2–6 + MLflow + salvamento de modelos
│   │   └── servico_avaliacao.py     # Avaliação de fornecedor individual ou em lote
│   │
│   ├── adaptadores/                 # Implementações concretas das portas
│   │   ├── entrada/
│   │   │   ├── leitor_kaggle.py     # Lê data/raw/data.csv (dataset Kaggle ESG)
│   │   │   └── leitor_fornecedores_pandas.py  # Lê planilhas XLSX/CSV enviadas pelo usuário
│   │   └── saida/
│   │       ├── repositorio_modelo_joblib.py   # Salva/carrega modelos com joblib
│   │       └── csv_resultado_adapter.py       # Exporta resultados para CSV
│   │
│   ├── infraestrutura/              # Detalhes técnicos (banco, auth, config, log)
│   │   ├── banco_dados.py           # Engine SQLAlchemy, SessaoLocal, obter_sessao()
│   │   ├── modelos_banco.py         # ORM: UsuarioBanco, FornecedorBanco, AvaliacaoBanco,
│   │   │                            #      PlanoAcaoBanco, ExperimentoMLBanco
│   │   ├── inicializacao_banco.py   # semear_banco(): migrações + seeds + views
│   │   ├── configuracoes.py         # Pydantic Settings (lê .env)
│   │   ├── seguranca.py             # JWT + bcrypt
│   │   └── registro_log.py          # structlog (JSON em produção, console em dev)
│   │
│   └── interfaces/
│       └── api/
│           ├── principal.py         # FastAPI app, lifespan, CORS, roteadores
│           ├── rotas_auth.py        # POST /auth/registrar|login, GET /auth/me
│           ├── rotas_enterprise.py  # Todos os endpoints de negócio
│           ├── esquemas.py          # Pydantic schemas de entrada/saída
│           ├── esquemas_auth.py     # Schemas de autenticação
│           └── dependencias_auth.py # obter_usuario_atual(), exigir_perfil()
│
├── frontend/
│   └── frontend/
│       ├── app.py                   # Aplicação Streamlit completa (SPA multi-página)
│       └── cliente_api.py           # ClienteApiESG: wrapper HTTP para todos os endpoints
│
├── airflow/
│   └── airflow/
│       └── dags/
│           └── dag_treinamento_esg.py  # DAG: re-treina modelos diariamente às 02:00
│
├── data/
│   ├── raw/
│   │   └── data.csv                 # Dataset Kaggle ESG (necessário para treinar)
│   └── amostras/
│       └── fornecedores_esg_simulado.csv  # Amostra para testes de importação em lote
│
├── docs/
│   ├── schema_banco_de_dados.md     # Schema completo do banco + queries do Dashboard
│   ├── instalacao.md                # Guia detalhado de instalação
│   ├── arquitetura_esg_nexus.html   # Diagrama interativo de arquitetura
│   └── diagrama_arquitetura_esg_nexus.html  # Diagrama alternativo
│
├── Dockerfile                       # Multi-stage build (builder + produção)
├── docker-compose.yml               # Orquestração de todos os serviços
├── docker-entrypoint.sh             # Corrige permissões de bind mounts e usa gosu
├── requirements.txt                 # Dependências Python
├── pyproject.toml                   # Metadados do projeto + mypy + ruff
├── treinar_modelo.py                # Script manual de treinamento (CRISP-DM)
├── inicializar_banco.py             # Atalho: python inicializar_banco.py
├── iniciar_backend.py               # Atalho: python iniciar_backend.py
├── testar_funcionalidades.py        # Smoke tests de integração
├── deploy_gcp.sh                    # Deploy automatizado em VM GCP Debian
├── scripts_inicializar_banco_linux.sh
└── scripts_inicializar_banco_windows.bat
```

---

## Módulos e Componentes

### Backend — FastAPI

**Arquivo de entrada:** `esg_ml/interfaces/api/principal.py`

- Inicializa o app FastAPI com lifespan que chama `semear_banco()` no startup
- Registra middleware CORS (origens: `localhost:8501`, `localhost:3000`, `localhost:5173`)
- Inclui dois roteadores: `rotas_auth` (prefixo `/auth`) e `rotas_enterprise` (sem prefixo)
- Expõe `GET /saude` que verifica se os modelos ML estão carregados em disco

**Autenticação:** JWT Bearer Token via `python-jose`. Senhas hasheadas com `bcrypt`. Tokens expiram em 480 minutos (configurável via `MINUTOS_EXPIRACAO_TOKEN`).

**Logging:** `structlog` com saída JSON estruturado em produção e colorido no desenvolvimento.

---

### Frontend — Streamlit

**Arquivo:** `frontend/frontend/app.py`

SPA (Single Page Application) com navegação lateral por perfil. Páginas disponíveis:

| Página | Perfis com acesso |
|--------|------------------|
| Dashboard Executivo ESG | administrador, gerente_esg, analista_esg |
| Dashboard Estatístico | administrador, gerente_esg, analista_esg, cientista_dados |
| Dashboard de Machine Learning | administrador, cientista_dados |
| Importação e Classificação em Lote | todos exceto cientista_dados |
| Fornecedores | todos exceto cientista_dados |
| Classificação e Explicabilidade | todos |
| Operação ML/Airflow/MLflow | administrador, cientista_dados |

**Cliente HTTP:** `frontend/frontend/cliente_api.py` — classe `ClienteApiESG` que encapsula todas as chamadas à API com tratamento de erros (`ErroAPI`). A URL base é lida de `ESG_NEXUS_API_URL` (padrão: `http://localhost:8000`).

---

### Machine Learning Pipeline

O pipeline segue o modelo **CRISP-DM** (Cross Industry Standard Process for Data Mining), fases 2 a 6:

| Fase | Descrição | Arquivo |
|------|-----------|---------|
| 2 — Entendimento dos dados | Carrega `data/raw/data.csv` (Kaggle ESG) | `adaptadores/entrada/leitor_kaggle.py` |
| 3 — Preparação dos dados | Limpeza, normalização, cálculo de pesos por setor, enriquecimento | `dominio/servicos/preprocessamento.py`, `feature_engineering.py` |
| 4 — Modelagem | GridSearchCV 5-fold: KNN k=3..21, RF com 24 combinações | `dominio/servicos/treinamento.py` |
| 5 — Avaliação | Critério: acc ≥ 93%, F1-Medium ≥ 88% | `dominio/servicos/avaliacao.py` |
| 6 — Implantação | Salva `modelo_knn.joblib`, `modelo_rf.joblib`, `config.joblib` em `artifacts/` | `adaptadores/saida/repositorio_modelo_joblib.py` |

**Variável-alvo:** `total_level` (`High` / `Medium`) derivada do dataset Kaggle.

**Features de entrada:**
- `environment_score`, `social_score`, `governance_score` (0–1000)
- `industry_enc` (setor codificado com LabelEncoder)

**Pesos por setor (feature engineering):** calculados como inverso da média normalizada do score por pilar — setores com menor desempenho ambiental recebem maior peso ambiental, incentivando a melhoria.

**Artefatos salvos em `artifacts/`:**

```
artifacts/
├── modelo_knn.joblib        # Modelo KNN + metadados (LabelEncoder, features, base_ref)
├── modelo_rf.joblib         # Modelo RF + metadados (LabelEncoder, features, importâncias)
├── config.joblib            # Configuração geral (pesos, benchmark por setor, mapa de maturidade)
├── bronze/
│   ├── base_processada.csv  # Dataset processado (CRISP-DM Fase 3)
│   └── pesos_por_industria.csv  # Pesos E/S/G calculados por setor
└── processado/
    ├── analise_exploratoria.png  # Gráficos EDA gerados no treino
    └── avaliacao_modelos.png     # Matrizes de confusão + feature importance
```

**Como treinar manualmente:**

```bash
# Dentro do container trainer (Docker Compose)
docker compose --profile training run trainer

# Ou localmente com o venv ativado
python treinar_modelo.py
```

---

### Banco de Dados — PostgreSQL / NeonDB

O projeto usa **NeonDB** (PostgreSQL serverless na AWS) como banco de dados padrão. A string de conexão completa fica em `DATABASE_URL` no arquivo `.env`.

**5 tabelas principais:**

| Tabela | Descrição |
|--------|-----------|
| `usuarios` | Contas de acesso à plataforma |
| `fornecedores` | Cadastro de fornecedores com scores ESG |
| `avaliacoes_esg` | Histórico de diagnósticos (uma linha por avaliação) |
| `planos_acao` | Itens do plano de melhoria por pilar ESG |
| `experimentos_ml` | Métricas de cada execução de treinamento |

**View:** `vw_fornecedores_classificacoes` — lista fornecedores com sua classificação ESG mais recente e planos de ação, usando `LATERAL JOIN` para eficiência.

**Migrações automáticas no startup:**
1. `_migrar_experimentos_ml()` — adiciona colunas `precision`/`recall` se ausentes
2. `_migrar_identity()` — converte PKs `SERIAL` → `GENERATED BY DEFAULT AS IDENTITY`
3. `criar_tabelas()` — `CREATE TABLE IF NOT EXISTS` via SQLAlchemy
4. `_criar_view_fornecedores_classificacoes()` — `CREATE OR REPLACE VIEW`

Para documentação detalhada do schema, ver [docs/schema_banco_de_dados.md](docs/schema_banco_de_dados.md).

---

### MLflow — Rastreamento de Experimentos

O MLflow registra automaticamente cada execução de treinamento:

- **Experimento:** `esg-nexus-fornecedores` (configurável via `MLFLOW_EXPERIMENTO`)
- **Runs registradas:** `KNN_GridSearch` e `RandomForest_GridSearch`
- **Params:** hiperparâmetros do GridSearchCV
- **Metrics:** acurácia, F1-score, precisão, recall (KNN e RF)
- **Artifacts:** gráficos EDA, matrizes de confusão, arquivos `.joblib`

**UI:** `http://localhost:5000` (ou porta configurada)

O Tracking URI é definido via `MLFLOW_TRACKING_URI` no `.env`. Em produção (GCP), aponta para o IP público da VM.

---

### Apache Airflow — Orquestração

**DAG:** `dag_treinamento_esg`  
**Arquivo:** `airflow/airflow/dags/dag_treinamento_esg.py`  
**Schedule:** diariamente às `02:00` UTC

**Tarefas (tasks):**

```
verificar_api  ──►  treinar_modelo_esg
```

1. `verificar_api`: chama `GET /saude` e verifica se KNN e RF estão carregados
2. `treinar_modelo_esg`: autentica via `POST /auth/login` (conta de serviço `ESG_ADMIN_EMAIL`) e chama `POST /treinar`

**UI:** `http://localhost:8080` — login: `admin` / `admin`

---

## API — Endpoints

Base URL: `http://localhost:8000`  
Documentação interativa: `http://localhost:8000/docs`

### Autenticação (`/auth`)

| Método | Rota | Acesso | Descrição |
|--------|------|--------|-----------|
| `POST` | `/auth/registrar` | Público | Cria novo usuário |
| `POST` | `/auth/login` | Público | Retorna JWT Bearer Token |
| `GET` | `/auth/me` | Autenticado | Dados do usuário logado |

### Saúde

| Método | Rota | Acesso | Descrição |
|--------|------|--------|-----------|
| `GET` | `/saude` | Público | Status da API e modelos ML carregados |

### Fornecedores

| Método | Rota | Acesso | Descrição |
|--------|------|--------|-----------|
| `POST` | `/fornecedores` | Autenticado | Cadastra novo fornecedor |
| `GET` | `/fornecedores` | Autenticado | Lista todos os fornecedores |
| `GET` | `/fornecedores/{id}/historico` | Autenticado | Histórico de avaliações do fornecedor |
| `GET` | `/fornecedores/{id}/plano-acao` | Autenticado | Plano de ação mais recente |
| `GET` | `/avaliacoes/{id}/plano-acao` | Autenticado | Plano de ação de avaliação específica |

### Classificação e Avaliação

| Método | Rota | Acesso | Descrição |
|--------|------|--------|-----------|
| `POST` | `/classificar` | Autenticado | Classifica um fornecedor (JSON) |
| `POST` | `/classificar/lote` | Autenticado | Classifica múltiplos fornecedores (JSON array) |
| `POST` | `/avaliar/upload` | Autenticado | Classifica fornecedores via upload de planilha XLSX/CSV |
| `POST` | `/explicabilidade` | Autenticado | Retorna explicação SHAP-like da classificação |

### Machine Learning

| Método | Rota | Acesso | Descrição |
|--------|------|--------|-----------|
| `POST` | `/treinar` | `administrador`, `cientista_dados` | Executa CRISP-DM Fases 2–6 e salva modelos |

### Dashboards

| Método | Rota | Acesso | Descrição |
|--------|------|--------|-----------|
| `GET` | `/dashboard/executivo` | Autenticado | KPIs, top risco, melhores fornecedores, view classificações |
| `GET` | `/dashboard/ml` | `administrador`, `cientista_dados` | Métricas ML, comparativo RF × KNN, feature importance |
| `GET` | `/dashboard/estatistico` | Autenticado | Estatísticas descritivas, correlações, séries temporais |

O endpoint `/dashboard/estatistico` aceita filtros via query string: `data_inicio`, `data_fim`, `setor`, `maturidade`.

---

## Perfis de Usuário e Controle de Acesso

| Perfil | Descrição | Acesso à API |
|--------|-----------|--------------|
| `administrador` | Acesso total | Todos os endpoints |
| `gerente_esg` | Gestão de fornecedores e dashboards | Fornecedores, classificação, dashboards executivo e estatístico |
| `analista_esg` | Análise e classificação | Idem gerente_esg |
| `operador_esg` | Operação de importação | Fornecedores, classificação, importação em lote |
| `cientista_dados` | ML e análise técnica | Treinamento, dashboards ML e estatístico, explicabilidade |

O **administrador padrão** é criado automaticamente no primeiro startup com as credenciais definidas em `USUARIO_ADMIN_EMAIL` e `USUARIO_ADMIN_SENHA` no `.env`.

---

## Variáveis de Ambiente

Copie `.env.example` para `.env` e ajuste os valores:

```env
# ── Banco de dados ────────────────────────────────────────────────────────────
# NeonDB (PostgreSQL serverless — AWS us-east-1)
DATABASE_URL=postgresql+psycopg://usuario:senha@host/banco?sslmode=require&channel_binding=require

# ── Segurança JWT ─────────────────────────────────────────────────────────────
SEGREDO_JWT=altere-este-segredo-em-producao          # use openssl rand -hex 32
ALGORITMO_JWT=HS256
MINUTOS_EXPIRACAO_TOKEN=480                          # 8 horas

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5000            # ou IP público em produção
MLFLOW_EXPERIMENTO=esg-nexus-fornecedores

# ── Frontend ──────────────────────────────────────────────────────────────────
ESG_NEXUS_API_URL=http://localhost:8000              # http://api:8000 dentro do Docker

# ── Administrador padrão (criado automaticamente no primeiro startup) ─────────
USUARIO_ADMIN_EMAIL=admin@esgnexus.example
USUARIO_ADMIN_SENHA=admin123                         # ALTERE EM PRODUÇÃO
USUARIO_ADMIN_NOME=Administrador ESG Nexus

# ── Conta de serviço do Airflow ───────────────────────────────────────────────
ESG_ADMIN_EMAIL=admin@esg.example
ESG_ADMIN_SENHA=admin@ESG2026!
```

> **Atenção de segurança:** Nunca comite o arquivo `.env` com credenciais reais no repositório. O arquivo `.env.example` contém apenas valores de exemplo.

---

## Instalação

### Pré-requisitos

| Ferramenta | Versão mínima | Verificação |
|------------|---------------|-------------|
| Git | qualquer | `git --version` |
| Docker | 24+ | `docker --version` |
| Docker Compose | 2.x | `docker compose version` |
| Python | 3.11 (instalação local) | `python --version` |

---

### Opção 1 — Docker Compose (recomendado)

Sobe todos os serviços em containers com um único comando. Ideal para desenvolvimento e produção em servidor.

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/ESGNexus.git
cd ESGNexus

# 2. Configure o ambiente
cp .env.example .env
# Edite .env e ajuste DATABASE_URL e SEGREDO_JWT antes de prosseguir

# 3. Suba todos os serviços
docker compose up --build -d

# 4. Acompanhe os logs
docker compose logs -f api

# 5. Verifique a saúde da API
curl http://localhost:8000/saude
```

**Serviços disponíveis após a subida:**

| Serviço | URL | Credenciais padrão |
|---------|-----|-------------------|
| API (FastAPI) | http://localhost:8000 | — |
| Docs interativos (Swagger) | http://localhost:8000/docs | — |
| Frontend (Streamlit) | http://localhost:8501 | admin@esgnexus.example / admin123 |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | admin / admin |

> O banco de dados (NeonDB) é externo e gerenciado. Tabelas e views são criadas automaticamente no startup da API.

**Executar o treinamento inicial dos modelos:**

```bash
# O modelo precisa ser treinado antes de usar /classificar
docker compose --profile training run --rm trainer
```

**Parar os serviços:**

```bash
docker compose down                  # para e remove containers (dados preservados)
docker compose down --remove-orphans # também remove containers órfãos
docker network prune -f              # limpa redes obsoletas (usar se houver erro de DNS)
```

**Atualizar após `git pull`:**

```bash
git pull
docker compose down --remove-orphans
docker network prune -f
docker compose build
docker compose up -d
```

---

### Opção 2 — Instalação local (desenvolvimento)

#### 1. Clone e configure

```bash
git clone https://github.com/seu-usuario/ESGNexus.git
cd ESGNexus
cp .env.example .env
# Edite .env com sua DATABASE_URL e SEGREDO_JWT
```

#### 2. Ambiente virtual Python

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Debian/Ubuntu** — instale dependências de sistema antes do pip:

```bash
sudo apt-get install -y libpq-dev build-essential python3.11-dev
```

#### 3. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
# ou via pyproject.toml
pip install -e .
```

#### 4. Inicialize o banco de dados

```bash
# Cria tabelas, views e usuário administrador padrão
python inicializar_banco.py

# Para recriar do zero (APAGA TODOS OS DADOS):
python -m esg_ml.infraestrutura.inicializacao_banco --recriar
```

#### 5. Treine os modelos ML

```bash
# Necessário antes de usar /classificar
# O arquivo data/raw/data.csv (Kaggle ESG) deve estar presente
python treinar_modelo.py
```

#### 6. Inicie o backend

```bash
uvicorn esg_ml.interfaces.api.principal:app --reload --host 0.0.0.0 --port 8000
# ou atalho:
python iniciar_backend.py
```

#### 7. Inicie o frontend (terminal separado)

```bash
# Com ESG_NEXUS_API_URL=http://localhost:8000 no .env
streamlit run frontend/frontend/app.py --server.port 8501
```

#### 8. MLflow (terminal separado, opcional)

```bash
mlflow ui --host 0.0.0.0 --port 5000 --default-artifact-root ./artifacts
```

#### 9. Apache Airflow (ambiente virtual separado, opcional)

O Airflow requer um venv isolado por incompatibilidade de dependências:

```bash
python -m venv .venv-airflow
source .venv-airflow/bin/activate

AIRFLOW_VERSION=2.9.3
PYTHON_VERSION=3.11
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://usuario:senha@host/banco"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/airflow/dags
export PYTHONPATH=$(pwd)   # necessário para importar esg_ml no DAG

airflow db migrate
airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname Admin \
  --role Admin --email admin@example.com

# Em terminais separados:
airflow webserver --port 8080
airflow scheduler
```

---

### Opção 3 — Máquina Virtual / Servidor Linux

Para deploy em VM (GCP, AWS EC2, DigitalOcean, etc.) com Debian/Ubuntu, use o script automatizado:

```bash
# Clone na VM
git clone https://github.com/seu-usuario/ESGNexus.git /opt/esgnexus
cd /opt/esgnexus

# Configure o .env ANTES de executar o deploy
cp .env.example .env
nano .env   # defina DATABASE_URL, SEGREDO_JWT, USUARIO_ADMIN_SENHA

# Execute o script de deploy
chmod +x deploy_gcp.sh
./deploy_gcp.sh
```

O script realiza automaticamente:

1. Instala dependências do sistema (`libpq-dev`, `ca-certificates`, `python3.11-venv`, etc.)
2. Cria usuário de serviço `esgnexus` sem shell (segurança)
3. Cria o venv Python e instala `requirements.txt`
4. Inicializa tabelas no NeonDB
5. Cria e habilita serviços **systemd** (`esgnexus-api`, `esgnexus-frontend`, `esgnexus-mlflow`)

**Gerenciamento dos serviços systemd:**

```bash
# Status
sudo systemctl status esgnexus-api esgnexus-frontend esgnexus-mlflow

# Logs em tempo real
journalctl -fu esgnexus-api

# Reiniciar após atualização
git pull
sudo systemctl restart esgnexus-api esgnexus-frontend esgnexus-mlflow
```

**Firewall — portas necessárias (GCP/AWS Security Groups):**

| Porta | Serviço | Protocolo |
|-------|---------|-----------|
| 8000 | API FastAPI | TCP |
| 8501 | Frontend Streamlit | TCP |
| 5000 | MLflow UI | TCP |
| 8080 | Airflow UI | TCP |

**Acesso externo ao frontend:**  
Configure `ESG_NEXUS_API_URL=http://<IP-PUBLICO-DA-VM>:8000` no `.env` do servidor para que o Streamlit conecte na API corretamente.

---

## Primeiro Treinamento do Modelo

Os modelos ML precisam ser treinados antes de qualquer classificação. O arquivo `data/raw/data.csv` (dataset Kaggle ESG Enterprise) deve estar presente.

```bash
# Docker Compose
docker compose --profile training run --rm trainer

# Local
python treinar_modelo.py

# Via API (usuário administrador ou cientista_dados)
curl -X POST http://localhost:8000/treinar \
  -H "Authorization: Bearer <seu-token>" \
  -H "Content-Type: application/json"
```

O pipeline leva tipicamente 2–5 minutos. Ao final, os seguintes arquivos serão criados em `artifacts/`:

```
artifacts/
├── modelo_knn.joblib
├── modelo_rf.joblib
├── config.joblib
├── bronze/base_processada.csv
├── bronze/pesos_por_industria.csv
└── processado/analise_exploratoria.png
```

As métricas são salvas automaticamente na tabela `experimentos_ml` do banco de dados e ficam visíveis no menu **Dashboard de Machine Learning** do frontend.

---

## Atualização em Produção

Sequência recomendada após `git pull` em ambiente Docker:

```bash
# 1. Baixar alterações
git pull

# 2. Parar containers e limpar rede (evita problemas de DNS interno)
docker compose down --remove-orphans
docker network prune -f

# 3. Reconstruir imagens
docker compose build

# 4. Subir novamente
docker compose up -d

# 5. Verificar saúde
docker compose ps
docker compose logs -f api
curl http://localhost:8000/saude
```

> As migrações de banco de dados (novas colunas, conversão para IDENTITY, views) são executadas **automaticamente** no startup da API — não é necessário rodar scripts manuais.

---

## Solução de Problemas

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| `Failed to resolve 'api'` no frontend | Rede Docker obsoleta após rebuild | `docker compose down --remove-orphans && docker network prune -f && docker compose up -d` |
| `function round(double precision, integer) does not exist` | Cast ausente para `NUMERIC` na view | Atualizar para a versão mais recente (corrigido em `inicializacao_banco.py`) |
| `column a.w_e does not exist` | Coluna criada com case-sensitive `"w_E"` | Usar `a."w_E" AS w_e` na view (corrigido) |
| `SSL connection required` | `sslmode` ausente na `DATABASE_URL` | Adicionar `?sslmode=require&channel_binding=require` ao final da URL |
| `ModuleNotFoundError: psycopg` | Driver não instalado | `pip install psycopg[c]>=3.2` (não `psycopg2`) |
| `Application startup failed` | Erro na criação da view no NeonDB | Verificar logs: `docker compose logs api` |
| Modelos não carregados (`/saude` retorna `false`) | Treino não executado | Rodar `python treinar_modelo.py` ou `POST /treinar` |
| Airflow `DAG import error` | `esg_ml` não no PYTHONPATH | `export PYTHONPATH=/opt/esgnexus` antes de iniciar o scheduler |
| `libpq.so not found` | `libpq5` ausente no sistema | `sudo apt-get install libpq5 libpq-dev` |
| Frontend mostra `Falha no login` | `ESG_NEXUS_API_URL` errada | Verificar a URL no `.env` e no campo "URL da API" na sidebar |
| Erro 403 em endpoint | Perfil sem permissão | Verificar perfil do usuário em `GET /auth/me` |

---

## Documentação Adicional

| Documento | Descrição |
|-----------|-----------|
| [docs/schema_banco_de_dados.md](docs/schema_banco_de_dados.md) | Schema completo do banco de dados: todas as tabelas, colunas, relacionamentos, view `vw_fornecedores_classificacoes` e todas as consultas SQL utilizadas nos dashboards |
| [docs/instalacao.md](docs/instalacao.md) | Guia detalhado de instalação para Docker, local e GCP |
| [docs/arquitetura_esg_nexus.html](docs/arquitetura_esg_nexus.html) | Diagrama interativo de arquitetura (abrir no navegador) |
| [docs/diagrama_arquitetura_esg_nexus.html](docs/diagrama_arquitetura_esg_nexus.html) | Diagrama alternativo de arquitetura (abrir no navegador) |

---

## Licença

Projeto desenvolvido para fins educacionais e corporativos. Consulte o administrador do repositório para informações sobre licenciamento.
