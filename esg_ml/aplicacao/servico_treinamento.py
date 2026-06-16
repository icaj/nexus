# esg_ml/aplicacao/servico_treinamento.py
# Caso de uso: Treinar modelos ESG — CRISP-DM Fases 2–6
# Mantém infraestrutura Enterprise (banco + MLflow) com núcleo nexus_v2

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from esg_ml.adaptadores.entrada.leitor_kaggle import LeitorKaggle
from esg_ml.adaptadores.saida.repositorio_modelo_joblib import RepositorioModeloJoblib
from esg_ml.dominio.servicos.preprocessamento import preprocessar
from esg_ml.dominio.servicos.feature_engineering import (
    calcular_pesos_por_industria, enriquecer_dataframe)
from esg_ml.dominio.servicos.treinamento import executar_treino
from esg_ml.dominio.servicos.avaliacao import avaliar_modelos, ModeloInsuficienteError
from esg_ml.infraestrutura.configuracoes import Configuracoes
from esg_ml.infraestrutura.registro_log import obter_registrador

registrador = obter_registrador(__name__)
conf = Configuracoes()


def _salvar_graficos(resultado: dict) -> list:
    pasta = str(conf.pasta_saida)
    os.makedirs(pasta, exist_ok=True)

    paths = []

    df = resultado['df']
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    vc = df['total_level'].value_counts()
    axes[0].pie(vc.values, labels=vc.index, colors=['#2ECC71','#E74C3C'],
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.55, edgecolor='white'))
    axes[0].set_title('Distribuição de Maturidade', fontweight='bold')
    top = df.groupby('industry')['total_score'].mean().sort_values(ascending=False).head(12)
    axes[1].barh(range(len(top)), top.values,
                 color=['#2ECC71' if v >= 900 else '#E74C3C' for v in top.values])
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top.index, fontsize=8)
    axes[1].set_title('Score Médio por Setor (Top 12)', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(pasta, 'analise_exploratoria.png')
    plt.savefig(p, dpi=140, bbox_inches='tight'); plt.close()
    paths.append(p)

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle('Avaliação dos Modelos ML — ESG', fontsize=14, fontweight='bold')
    le_target = resultado['le_target']
    for ax, y_pred, titulo, cmap in [
        (axes[0], resultado['y_pred_knn'], 'KNN', 'Blues'),
        (axes[1], resultado['y_pred_rf'],  'Random Forest', 'Greens')]:
        cm   = confusion_matrix(resultado['y_test'], y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(titulo, fontweight='bold')
    feats  = [f[0] for f in resultado['importancias']]
    values = [f[1] for f in resultado['importancias']]
    cores  = ['#E67E22' if v<0.05 else ('#2980B9' if v<0.30 else '#2ECC71') for v in values]
    axes[2].barh(range(len(feats)), values, color=cores, edgecolor='white', height=0.6)
    axes[2].set_yticks(range(len(feats)))
    axes[2].set_yticklabels(feats, fontsize=9)
    axes[2].set_title('Importância das Variáveis\n(Random Forest)', fontweight='bold')
    axes[2].set_xlabel('Gini'); axes[2].set_xlim(0, 0.85)
    for i,v in enumerate(values):
        axes[2].text(v+0.01, i, f'{v:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    p = os.path.join(pasta, 'avaliacao_modelos.png')
    plt.savefig(p, dpi=140, bbox_inches='tight'); plt.close()
    paths.append(p)
    return paths


class ServicoTreinamento:
    """
    Caso de uso de treino — mantém interface Enterprise mas usa núcleo nexus_v2:
    - Dados: ESG Enterprise (Kaggle) via LeitorKaggle
    - GridSearchCV: KNN k=3..21 + RF 24 combinações
    - Critério Fase 5: acc>=93%, F1-Medium>=88%
    - Salva: modelo_knn.joblib, modelo_rf.joblib, config.joblib
    """

    def __init__(self, repositorio: RepositorioModeloJoblib = None) -> None:
        self.repositorio = repositorio or RepositorioModeloJoblib(conf.diretorio_artefatos)

    def treinar(self, fornecedores=None) -> dict:
        """Executa CRISP-DM Fases 2–6. Dados do Kaggle; MLflow sempre registrado."""
        print('╔══════════════════════════════════════════════════════════╗')
        print('║  PIPELINE ESG — CRISP-DM + Arquitetura Hexagonal         ║')
        print('╚══════════════════════════════════════════════════════════╝')

        # Fase 2
        print('\n[FASE 2] Carregando dados...')
        caminho_dados = str(conf.diretorio_artefatos.parent / 'data' / 'raw' / 'data.csv')
        df_raw = LeitorKaggle().carregar(caminho_dados)
        print(f'  {len(df_raw)} empresas × {len(df_raw.columns)} colunas')

        # Fase 3
        print('\n[FASE 3] Pré-processamento e feature engineering...')
        df = preprocessar(df_raw)
        pesos_por_ind, pesos_global = calcular_pesos_por_industria(df)
        df = enriquecer_dataframe(df, pesos_por_ind, pesos_global)
        pasta_bronze = str(conf.pasta_bronze)
        os.makedirs(pasta_bronze, exist_ok=True)
        base_ref = df[['name','industry','environment_score','social_score',
                        'governance_score','total_score','score_ponderado',
                        'total_grade','total_level','maturidade',
                        'risco','impacto','quadrante']].copy()
        base_ref.to_csv(f'{pasta_bronze}/base_processada.csv', index=False)
        pd.DataFrame([
            {'industry': ind, 'w_E': p.w_E, 'w_S': p.w_S, 'w_G': p.w_G, 'fonte': p.fonte}
            for ind, p in pesos_por_ind.items()
        ]).to_csv(f'{pasta_bronze}/pesos_por_industria.csv', index=False)
        print(f'  {df["industry"].nunique()} setores | High: {(df["total_level"]=="High").mean():.1%}')

        # Fase 4
        print('\n[FASE 4] Treinando modelos (GridSearchCV 5-fold)...')
        resultado = executar_treino(df)
        print(f'  KNN → {resultado["knn_params"]} | CV: {resultado["knn_cv_score"]:.2%}')
        print(f'  RF  → {resultado["rf_params"]}  | CV: {resultado["rf_cv_score"]:.2%}')

        # Fase 5
        print('\n[FASE 5] Avaliando modelos (acc>=93%, F1-Medium>=88%)...')
        try:
            metricas = avaliar_modelos(resultado)
            print(f'  KNN aprovado → acc: {metricas["knn"]["accuracy"]:.2%} | F1-Med: {metricas["knn"]["f1_medium"]:.2%}')
            print(f'  RF  aprovado → acc: {metricas["rf"]["accuracy"]:.2%}  | F1-Med: {metricas["rf"]["f1_medium"]:.2%}')
        except ModeloInsuficienteError as e:
            registrador.error('modelo_insuficiente', erro=str(e))
            print(f'\n[FASE 5] REPROVADO:\n{e}')
            raise  # não usar sys.exit() — mata o uvicorn quando chamado via API

        graficos = _salvar_graficos(resultado)

        # Fase 6
        print('\n[FASE 6] Salvando modelos aprovados...')
        benchmark = df.groupby('industry')[
            ['environment_score','social_score','governance_score','total_score','score_ponderado']
        ].mean().round(1).to_dict()

        self.repositorio.salvar('modelo_knn', resultado['knn_model'], {
            'le_target':   resultado['le_target'],
            'le_industry': resultado['le_ind'],
            'features':    resultado['FEATURES'],
            'base_ref':    base_ref.to_dict('records'),
        })
        self.repositorio.salvar('modelo_rf', resultado['rf_model'], {
            'le_target':    resultado['le_target'],
            'le_industry':  resultado['le_ind'],
            'features':     resultado['FEATURES'],
            'importancias': resultado['importancias'],
            'base_ref':     base_ref.to_dict('records'),
        })
        self.repositorio.salvar('config', None, {
            'le_industry':      resultado['le_ind'],
            'le_target':        resultado['le_target'],
            'features':         resultado['FEATURES'],
            'benchmark':        benchmark,
            'mapa_maturidade':  {'High':'Avançado','Medium':'Iniciante'},
            'pesos_por_ind':    pesos_por_ind,
            'pesos_global':     pesos_global,
            'min_empresas_peso': 5,
        })

        self._registrar_mlflow(resultado, metricas, graficos)

        registrador.info('treino_concluido',
                         knn_acc=metricas['knn']['accuracy'],
                         rf_acc=metricas['rf']['accuracy'])

        print('\n╔══════════════════════════════════════════════════════════╗')
        print('║  PIPELINE CONCLUÍDO — modelos aprovados e salvos         ║')
        print('╚══════════════════════════════════════════════════════════╝')

        return {
            'knn_acuracia':  metricas['knn']['accuracy'],
            'knn_f1_medium': metricas['knn']['f1_medium'],
            'knn_precision': metricas['knn']['precision_macro'],
            'knn_recall':    metricas['knn']['recall_macro'],
            'rf_acuracia':   metricas['rf']['accuracy'],
            'rf_f1_medium':  metricas['rf']['f1_medium'],
            'rf_precision':  metricas['rf']['precision_macro'],
            'rf_recall':     metricas['rf']['recall_macro'],
            'knn_params':    str(resultado['knn_params']),
            'rf_params':     str(resultado['rf_params']),
        }

    def _registrar_mlflow(self, resultado: dict, metricas: dict, graficos: list) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(conf.mlflow_tracking_uri)
            mlflow.set_experiment(conf.mlflow_experimento)
            artefatos = conf.diretorio_artefatos
            for nome, params, met, joblib_nome in [
                ('KNN_GridSearch',          resultado['knn_params'], metricas['knn'], 'modelo_knn.joblib'),
                ('RandomForest_GridSearch', resultado['rf_params'],  metricas['rf'],  'modelo_rf.joblib'),
            ]:
                with mlflow.start_run(run_name=nome):
                    mlflow.log_params(params)
                    mlflow.log_metrics(met)
                    for g in graficos:
                        mlflow.log_artifact(g, 'graficos')
                    mlflow.log_artifact(str(artefatos / joblib_nome), 'modelo')
            print('  MLflow: experimentos registrados.')
        except Exception as e:
            registrador.warning('falha_registro_mlflow', erro=str(e))
            print(f'  [AVISO] MLflow não disponível: {e}')
