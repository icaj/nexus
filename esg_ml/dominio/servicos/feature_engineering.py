# esg_ml/dominio/servicos/feature_engineering.py
# CRISP-DM Fase 3 — Feature Engineering e métricas ESG (lógica pura)
import pandas as pd
from esg_ml.dominio.entidades.empresa import PesosSetor, Quadrante, Maturidade

FEATURES_SCORES   = ['environment_score','social_score','governance_score']
MIN_EMPRESAS_PESO = 5
MAPA_MATURIDADE   = Maturidade.MAPA
GRADE_MAP = {
    range(0,    600):  ('Abaixo do mínimo','D'),
    range(600,  750):  ('Baixo','B'),
    range(750,  900):  ('Baixo-Médio','BB'),
    range(900,  1200): ('Médio-Alto','BBB'),
    range(1200, 1800): ('Alto','A'),
    range(1800, 3001): ('Excelente','AA+'),
}

def mapear_grade(total_score: int) -> tuple:
    for r,(level,grade) in GRADE_MAP.items():
        if total_score in r: return level, grade
    return ('Desconhecido','?')

def calcular_pesos_globais(df: pd.DataFrame) -> dict:
    corr = df[FEATURES_SCORES].corrwith(df['total_score']).clip(lower=0)
    return (corr/corr.sum()).to_dict()

def calcular_pesos_por_industria(df: pd.DataFrame) -> tuple:
    pesos_global = calcular_pesos_globais(df)
    pesos = {}
    for ind, grupo in df.groupby('industry'):
        n = len(grupo)
        if n >= MIN_EMPRESAS_PESO:
            corr = grupo[FEATURES_SCORES].corrwith(grupo['total_score']).clip(lower=0)
            soma = corr.sum()
            w    = (corr/soma).to_dict() if soma > 0 else pesos_global
            fonte = f'empírico (n={n})'
        else:
            w = pesos_global
            fonte = f'fallback (n={n}<{MIN_EMPRESAS_PESO})'
        pesos[ind] = PesosSetor(
            w_E=round(w['environment_score'],4),
            w_S=round(w['social_score'],4),
            w_G=round(w['governance_score'],4),
            fonte=fonte)
    return pesos, pesos_global

def calcular_score_ponderado(env:int, soc:int, gov:int, pesos:PesosSetor) -> float:
    return round(pesos.w_E*env + pesos.w_S*soc + pesos.w_G*gov, 1)

def calcular_risco(score_ponderado: float) -> float:
    return round((1000-score_ponderado)/1000*100, 1)

def calcular_impacto_df(df: pd.DataFrame) -> pd.Series:
    return df.groupby('industry')['score_ponderado'].rank(pct=True).mul(100).round(1)

def calcular_impacto_empresa(score_ponderado: float, industry: str, benchmark: dict) -> float:
    media  = benchmark.get(industry, 355.0)
    desvio = 110.0
    z = (score_ponderado - media) / desvio
    return round(max(0.0, min(100.0, 50+z*25)), 1)

def enriquecer_dataframe(df: pd.DataFrame, pesos_por_ind: dict, pesos_global: dict) -> pd.DataFrame:
    df = df.copy()
    def _sp(row):
        p  = pesos_por_ind.get(row['industry'])
        we = p.w_E if p else pesos_global['environment_score']
        ws = p.w_S if p else pesos_global['social_score']
        wg = p.w_G if p else pesos_global['governance_score']
        return round(we*row['environment_score']+ws*row['social_score']+wg*row['governance_score'],1)
    df['score_ponderado'] = df.apply(_sp, axis=1)
    df['maturidade']      = df['total_level'].map(MAPA_MATURIDADE)
    df['risco']           = ((1000-df['score_ponderado'])/1000*100).round(1)
    df['impacto']         = calcular_impacto_df(df)
    df['quadrante']       = df.apply(lambda r: Quadrante.classificar(r['impacto'],r['risco']), axis=1)
    return df
