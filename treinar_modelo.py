#!/usr/bin/env python3
"""Entry point: treina os modelos ESG com dados Kaggle (CRISP-DM Fases 2–6)."""
from datetime import datetime, timezone

from esg_ml.aplicacao.servico_treinamento import ServicoTreinamento

if __name__ == '__main__':
    metricas = ServicoTreinamento().treinar()

    from esg_ml.infraestrutura.banco_dados import SessaoLocal
    from esg_ml.infraestrutura.modelos_banco import ExperimentoMLBanco
    with SessaoLocal() as sessao:
        exp = ExperimentoMLBanco(
            nome_execucao=f"treino_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            knn_acuracia=metricas.get('knn_acuracia', 0.0),
            knn_f1_medium=metricas.get('knn_f1_medium', 0.0),
            knn_precision=metricas.get('knn_precision'),
            knn_recall=metricas.get('knn_recall'),
            rf_acuracia=metricas.get('rf_acuracia', 0.0),
            rf_f1_medium=metricas.get('rf_f1_medium', 0.0),
            rf_precision=metricas.get('rf_precision'),
            rf_recall=metricas.get('rf_recall'),
            knn_params=str(metricas.get('knn_params', '')),
            rf_params=str(metricas.get('rf_params', '')),
        )
        sessao.add(exp)
        sessao.commit()
        print(f'\nMétricas salvas no banco (experimento id={exp.id})')

    print('\nMétricas finais:')
    for k, v in metricas.items():
        print(f'  {k}: {v}')
