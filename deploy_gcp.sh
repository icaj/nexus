#!/usr/bin/env bash
# deploy_gcp.sh — Instalação da aplicação ESG Nexus Enterprise
# em instância GCP com Debian 11/12 (sem Docker).
#
# Uso:
#   chmod +x deploy_gcp.sh
#   ./deploy_gcp.sh
#
# Pré-requisito: executar como usuário com sudo (não como root direto).

set -euo pipefail

APP_DIR="/opt/esgnexus"
APP_USER="esgnexus"
PYTHON="python3.11"

echo "=== [1/7] Atualizando pacotes do sistema ==="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libpq-dev \
    libpq5 \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip

echo "=== [2/7] Criando usuário de serviço ==="
if ! id "$APP_USER" &>/dev/null; then
    sudo useradd --system --no-create-home --shell /usr/sbin/nologin "$APP_USER"
fi

echo "=== [3/7] Clonando / atualizando repositório ==="
if [ ! -d "$APP_DIR/.git" ]; then
    sudo git clone https://github.com/seu-usuario/ESGNexus.git "$APP_DIR"
else
    sudo git -C "$APP_DIR" pull --ff-only
fi
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

echo "=== [4/7] Criando ambiente virtual e instalando dependências ==="
sudo -u "$APP_USER" "$PYTHON" -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "=== [5/7] Configurando variáveis de ambiente ==="
if [ ! -f "$APP_DIR/.env" ]; then
    sudo cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "ATENÇÃO: edite $APP_DIR/.env e ajuste SEGREDO_JWT antes de iniciar."
    echo "         A DATABASE_URL já aponta para o NeonDB."
fi

echo "=== [6/7] Inicializando banco de dados ==="
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    source .venv/bin/activate
    source .env
    export DATABASE_URL
    python -m esg_ml.infraestrutura.inicializacao_banco
"

echo "=== [7/7] Criando serviços systemd ==="

# ── API (FastAPI via uvicorn) ──────────────────────────────────────
sudo tee /etc/systemd/system/esgnexus-api.service > /dev/null <<EOF
[Unit]
Description=ESG Nexus Enterprise — API (FastAPI)
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn esg_ml.interfaces.api.principal:app --host 0.0.0.0 --port 8000 --workers 2
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ── Frontend (Streamlit) ───────────────────────────────────────────
sudo tee /etc/systemd/system/esgnexus-frontend.service > /dev/null <<EOF
[Unit]
Description=ESG Nexus Enterprise — Frontend (Streamlit)
After=esgnexus-api.service

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/streamlit run frontend_streamlit/app.py --server.address 0.0.0.0 --server.port 8501
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ── MLflow ────────────────────────────────────────────────────────
sudo tee /etc/systemd/system/esgnexus-mlflow.service > /dev/null <<EOF
[Unit]
Description=ESG Nexus Enterprise — MLflow UI
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/mlflow ui --host 0.0.0.0 --port 5000 --default-artifact-root $APP_DIR/artifacts
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable esgnexus-api esgnexus-frontend esgnexus-mlflow
sudo systemctl start  esgnexus-api esgnexus-frontend esgnexus-mlflow

echo ""
echo "=== Deploy concluído ==="
echo "  API       → http://$(hostname -I | awk '{print $1}'):8000"
echo "  Frontend  → http://$(hostname -I | awk '{print $1}'):8501"
echo "  MLflow    → http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Logs: journalctl -fu esgnexus-api"
