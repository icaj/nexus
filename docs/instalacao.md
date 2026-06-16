# Guia de Instalação — ESG Nexus Enterprise

Este guia cobre a instalação de todos os módulos da aplicação: **Backend (FastAPI)**, **Frontend (Streamlit)**, **NeonDB (PostgreSQL gerenciado)**, **MLflow** e **Apache Airflow**.

O banco de dados utilizado é o **NeonDB** (PostgreSQL serverless gerenciado na AWS) — não é necessário instalar ou manter um PostgreSQL local.

---

## Pré-requisitos

| Ferramenta | Versão mínima | Verificar |
|---|---|---|
| Python | 3.11 | `python --version` |
| Docker | 24+ | `docker --version` |
| Docker Compose | 2.x | `docker compose version` |
| Git | qualquer | `git --version` |

---

## Opção 1 — Docker Compose (recomendado para desenvolvimento)

Sobe todos os serviços de uma vez: API, Streamlit, MLflow e Airflow. O banco de dados é o NeonDB — não há container de PostgreSQL.

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ESGNexus.git
cd ESGNexus

# Copie e confira o arquivo de ambiente (DATABASE_URL já aponta para NeonDB)
cp .env.example .env

# Ajuste SEGREDO_JWT antes de subir em qualquer ambiente
# Suba todos os serviços
docker compose up --build -d

# Acompanhe os logs
docker compose logs -f
```

### Serviços e portas

| Serviço | URL | Credenciais padrão |
|---|---|---|
| API (FastAPI) | http://localhost:8000 | — |
| Docs interativos | http://localhost:8000/docs | — |
| Frontend (Streamlit) | http://localhost:8501 | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | admin / admin |
| NeonDB | gerenciado (externo) | ver `.env` |

### Parar os serviços

```bash
docker compose down        # para e remove os containers
docker compose down -v     # também remove volumes locais de artefatos
```

---

## Opção 2 — Instalação local (desenvolvimento)

### 1. Backend (FastAPI)

```bash
# Crie e ative o ambiente virtual
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Instale as dependências (psycopg[c] exige libpq-dev no sistema)
pip install -r requirements.txt
# ou via pyproject.toml
pip install -e .

# Configure as variáveis de ambiente
cp .env.example .env   # DATABASE_URL já aponta para NeonDB
```

> **Debian/Ubuntu:** antes do `pip install`, instale `libpq-dev` para compilar `psycopg[c]`:
> ```bash
> sudo apt-get install -y libpq-dev build-essential python3.11-dev
> ```

Inicie o servidor:

```bash
uvicorn esg_ml.interfaces.api.principal:app --reload --host 0.0.0.0 --port 8000
```

---

### 2. Frontend (Streamlit)

```bash
# Com o .venv ativado
streamlit run frontend_streamlit/app.py --server.port 8501
```

Variável de ambiente necessária (já no `.env`):

```env
ESG_NEXUS_API_URL=http://localhost:8000
```

---

### 3. NeonDB (banco de dados)

Nenhuma instalação necessária. O NeonDB é um serviço PostgreSQL gerenciado — a string de conexão já está em `.env.example`.

Para criar/recriar as tabelas:

```bash
python -m esg_ml.infraestrutura.inicializacao_banco
# ou forçar recriação:
python -m esg_ml.infraestrutura.inicializacao_banco --recriar
```

---

### 4. MLflow

```bash
# Com o .venv ativado
mlflow ui \
  --host 0.0.0.0 \
  --port 5000 \
  --default-artifact-root ./artifacts
```

> O MLflow usa a mesma `DATABASE_URL` do `.env` para persistir experimentos.

Acesse a UI em http://localhost:5000.

---

### 5. Apache Airflow

O Airflow é instalado em um ambiente virtual separado (dependências incompatíveis com o resto).

```bash
python -m venv .venv-airflow
source .venv-airflow/bin/activate   # Linux/macOS
# .venv-airflow\Scripts\activate    # Windows

AIRFLOW_VERSION=2.9.3
PYTHON_VERSION=3.11
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Aponte para o NeonDB (mesmo banco, schema separado pelo Airflow)
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="$DATABASE_URL"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/airflow/dags

airflow db migrate

airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

Terminais separados:

```bash
airflow webserver --port 8080
airflow scheduler
```

---

## Opção 3 — Deploy em instância GCP Debian (produção)

### Pré-requisitos na VM

- Debian 11 (Bullseye) ou 12 (Bookworm)
- Python 3.11 disponível (`sudo apt-get install python3.11`)
- Acesso SSH com sudo

### Script automatizado

```bash
# Na sua máquina local, copie o script para a VM
gcloud compute scp deploy_gcp.sh SEU_USUARIO@SUA_VM:/tmp/

# Ou clone diretamente na VM
git clone https://github.com/seu-usuario/ESGNexus.git /tmp/ESGNexus
chmod +x /tmp/ESGNexus/deploy_gcp.sh
/tmp/ESGNexus/deploy_gcp.sh
```

O script `deploy_gcp.sh` realiza automaticamente:

1. Instala dependências de sistema (`libpq-dev`, `ca-certificates`, `python3.11-venv`, etc.)
2. Cria usuário de serviço `esgnexus` (sem shell, sem home)
3. Clona ou atualiza o repositório em `/opt/esgnexus`
4. Cria o `.venv` e instala `requirements.txt`
5. Inicializa as tabelas no NeonDB
6. Cria e habilita serviços **systemd** para API, Frontend e MLflow

### Variáveis de ambiente na VM

Após o deploy, edite `/opt/esgnexus/.env` e defina pelo menos:

```env
SEGREDO_JWT=uma-chave-longa-e-aleatoria-para-producao
```

Reinicie os serviços após editar:

```bash
sudo systemctl restart esgnexus-api esgnexus-frontend esgnexus-mlflow
```

### Firewall GCP

Abra as portas necessárias no Console GCP → VPC network → Firewall rules:

| Porta | Serviço |
|---|---|
| 8000 | API (FastAPI) |
| 8501 | Frontend (Streamlit) |
| 5000 | MLflow UI |
| 8080 | Airflow UI |

### Gerenciamento dos serviços

```bash
# Status
sudo systemctl status esgnexus-api

# Logs em tempo real
journalctl -fu esgnexus-api

# Reiniciar
sudo systemctl restart esgnexus-api
```

---

## Ordem de inicialização recomendada (local)

```
1. NeonDB       → externo, sempre disponível
2. MLflow       → registra experimentos usados pela API
3. Backend (API)→ expõe endpoints consumidos pelo frontend e pelo Airflow
4. Frontend     → interface do usuário
5. Airflow      → orquestração de pipelines (opcional para desenvolvimento)
```

---

## Verificação rápida

```bash
# API health check
curl http://localhost:8000/health

# Docs interativos
xdg-open http://localhost:8000/docs   # Linux
start http://localhost:8000/docs      # Windows
open http://localhost:8000/docs       # macOS
```

---

## Solução de problemas comuns

| Problema | Causa provável | Solução |
|---|---|---|
| `SSL connection required` | `sslmode` ausente na URL | Verifique que a `DATABASE_URL` termina com `?sslmode=require&channel_binding=require` |
| `psycopg.errors.FeatureNotSupported` | Driver psycopg2 em vez de psycopg3 | Confirme que instalou `psycopg[c]>=3.2` (não `psycopg2`) |
| `ModuleNotFoundError: psycopg` | Dependências não instaladas | Rode `pip install -r requirements.txt` com o venv ativo |
| `connection refused` na porta 8000 | API não iniciou | Verifique `systemctl status esgnexus-api` / `docker compose ps` |
| Airflow `DAG import error` | Módulos do projeto não no `PYTHONPATH` | Exporte `PYTHONPATH=/opt/esgnexus` antes de iniciar o scheduler |
| MLflow `artifact location` vazio | Pasta `./artifacts` não existe | `mkdir -p artifacts` na raiz do projeto |
| Streamlit não conecta na API | `ESG_NEXUS_API_URL` errada | Confirme a variável apontando para `http://localhost:8000` (ou IP da VM) |
| `libpq.so not found` no build | `libpq5` não instalado | `sudo apt-get install libpq5 libpq-dev` |
