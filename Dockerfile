# ── Estágio de build ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Dependências de sistema necessárias para compilar extensões C do psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Estágio de produção ───────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

WORKDIR /app

# ca-certificates: necessário para conexões TLS com NeonDB e outros serviços em nuvem
# gosu: troca de usuário segura no entrypoint (mesma abordagem das imagens oficiais postgres/redis)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpq5 \
        gosu \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas os pacotes instalados (sem ferramentas de build)
COPY --from=builder /install /usr/local

# Usuário não-root para execução segura em produção (GCP / Cloud Run)
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home appuser

COPY --chown=appuser:appgroup . .

# Cria diretórios de dados com permissão correta na imagem
RUN mkdir -p artifacts/bronze artifacts/processado artifacts/resultados \
              data/raw data/bronze data/processado data/resultados \
    && chown -R appuser:appgroup artifacts data

# Entrypoint: corrige permissões dos bind mounts do host e faz drop para appuser via gosu
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Porta padrão da API; o Streamlit usa 8501
EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:$PATH"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "esg_ml.interfaces.api.principal:app", "--host", "0.0.0.0", "--port", "8000"]
