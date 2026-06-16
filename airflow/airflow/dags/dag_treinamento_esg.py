from __future__ import annotations

from datetime import datetime
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

# URL da API dentro da rede Docker; sobreposta por ESG_NEXUS_API_URL no .env
URL_API = os.getenv("ESG_NEXUS_API_URL", "http://api:8000")

# Credenciais da conta de serviço do Airflow — definir no .env do servidor
# ESG_ADMIN_EMAIL e ESG_ADMIN_SENHA devem pertencer a um usuário administrador
# ou cientista_dados criado previamente via POST /auth/registrar.
ADMIN_EMAIL = os.getenv("ESG_ADMIN_EMAIL", "admin@esg.example")
ADMIN_SENHA  = os.getenv("ESG_ADMIN_SENHA",  "admin@ESG2026!")

# Script bash: obtém JWT → chama /treinar com autenticação
_SCRIPT_TREINAR = f"""
set -e
echo "[1/2] Autenticando na API..."
TOKEN=$(curl -sf -X POST {URL_API}/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{{"email":"{ADMIN_EMAIL}","senha":"{ADMIN_SENHA}"}}' \\
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo "Token obtido."

echo "[2/2] Disparando treinamento ESG (CRISP-DM)..."
curl -sf -X POST {URL_API}/treinar \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  | python3 -m json.tool
echo "Treinamento concluído."
"""

with DAG(
    dag_id="dag_treinamento_esg",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    tags=["esg", "machine-learning", "fornecedores"],
    doc_md="""
## DAG Treinamento ESG

Re-treina os modelos KNN e Random Forest diariamente às 02:00.

**Pré-requisitos (primeira execução):**
1. Definir `ESG_ADMIN_EMAIL` e `ESG_ADMIN_SENHA` no `.env` do servidor
   (a conta é criada automaticamente no startup da API se não existir)
2. Garantir que o arquivo `data/raw/data.csv` (Kaggle ESG) esteja disponível

**Fluxo:** verificar_api → treinar_modelo_esg
""",
) as dag:

    verificar_api = BashOperator(
        task_id="verificar_api",
        bash_command=(
            f"curl --fail --silent {URL_API}/saude "
            "| python3 -c \""
            "import sys,json; d=json.load(sys.stdin); "
            "ok=all(d.values()); "
            "print('Modelos carregados:', d); "
            "exit(0 if ok else 1)"
            "\""
        ),
        doc_md="Verifica se a API está online **e** se os modelos ML estão carregados.",
    )

    treinar_modelo = BashOperator(
        task_id="treinar_modelo_esg",
        bash_command=_SCRIPT_TREINAR,
        execution_timeout=None,  # treinamento pode demorar vários minutos
        doc_md="Autentica com JWT e chama POST /treinar (CRISP-DM Fases 2–6).",
    )

    verificar_api >> treinar_modelo
