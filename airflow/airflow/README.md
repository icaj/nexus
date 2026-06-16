# Airflow

O Airflow orquestra o pipeline de treinamento e geração de dashboards.

## DAG

```text
dag_treinamento_esg
```

## Fluxo

```text
verificar_api
      ↓
treinar_modelo_esg
      ↓
gerar_dashboards_html
```

## Execução via Docker Compose

```bash
docker compose up airflow-webserver airflow-scheduler airflow-init
```

Interface:

```text
http://localhost:8080
```

Credenciais padrão:

```text
admin / admin
```
