# DAGs Airflow

Diretório com DAGs do Apache Airflow.

---

## DAG disponível

```text
dag_treinamento_esg.py
```

---

## Finalidade

A DAG representa o processo de treinamento recorrente do modelo ESG.

Fluxo esperado:

```text
Início
  ↓
Carregar dados
  ↓
Validar dados
  ↓
Treinar modelo
  ↓
Registrar métricas no MLflow
  ↓
Persistir artefato
  ↓
Fim
```

---

## Como usar

Copie este arquivo para a pasta configurada como `dags_folder` do Airflow.

Exemplo:

```bash
cp airflow/dags/dag_treinamento_esg.py ~/airflow/dags/
```

Depois acesse a UI do Airflow e habilite a DAG.

---

## Observação

A DAG foi incluída como referência arquitetural. Dependendo do ambiente, pode ser necessário ajustar paths, variáveis de ambiente e conexões.

