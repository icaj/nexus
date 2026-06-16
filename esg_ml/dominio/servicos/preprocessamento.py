# esg_ml/dominio/servicos/preprocessamento.py
# CRISP-DM Fase 3 — Limpeza dos dados (lógica pura, sem I/O)
import pandas as pd

COLUNAS_REMOVER = ['ticker','logo','weburl','last_processing_date','cik','currency','exchange']
FEATURES_SCORES = ['environment_score','social_score','governance_score']
NOMES_CANONICOS = {
    'Aerospace & Defense':           'Aerospace and Defense',
    'Hotels, Restaurants & Leisure': 'Hotels Restaurants and Leisure',
    'Metals & Mining':               'Metals and Mining',
}

def preprocessar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in COLUNAS_REMOVER if c in df.columns])
    df['industry'] = df['industry'].fillna('Unknown').str.strip().replace(NOMES_CANONICOS)
    soma = df[FEATURES_SCORES].sum(axis=1)
    if not (soma == df['total_score']).all():
        raise ValueError('Integridade violada: total_score != E + S + G')
    return df
