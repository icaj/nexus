# esg_ml/dominio/servicos/predicao.py
# CRISP-DM Fase 6 — Predição pura (sem I/O, sem efeitos colaterais)
from typing import List
from esg_ml.dominio.entidades.empresa import (
    Empresa, PesosSetor, Maturidade, Quadrante, PLANOS_ACAO, LIMIAR_CONFORMIDADE)
from esg_ml.dominio.entidades.diagnostico import DiagnosticoESG, ItemPlanoAcao
from esg_ml.dominio.servicos.feature_engineering import (
    calcular_score_ponderado, calcular_risco, calcular_impacto_empresa, mapear_grade)

def _encode_industria(industry: str, le_industry) -> int:
    try:    return le_industry.transform([industry])[0]
    except: return le_industry.transform(['Unknown'])[0]

def classificar_knn(vetor: list, knn_pack: dict) -> tuple:
    modelo    = knn_pack['modelo']
    le_target = knn_pack['le_target']
    base_ref  = knn_pack['base_ref']
    pred  = le_target.inverse_transform(modelo.predict([vetor]))[0]
    proba = modelo.predict_proba([vetor])[0]
    conf  = {c: round(p*100,1) for c,p in zip(le_target.classes_, proba)}
    dist, idx = modelo.named_steps['knn'].kneighbors(
        modelo.named_steps['scaler'].transform([vetor]), n_neighbors=3)
    vizinhos = [
        {'nome': base_ref[i]['name'], 'industry': base_ref[i]['industry'],
         'maturidade': base_ref[i]['maturidade'], 'total_score': base_ref[i]['total_score'],
         'distancia': round(d,2)}
        for d,i in zip(dist[0], idx[0])
    ]
    return pred, conf, vizinhos

def classificar_rf(vetor: list, rf_pack: dict) -> tuple:
    modelo    = rf_pack['modelo']
    le_target = rf_pack['le_target']
    pred  = le_target.inverse_transform(modelo.predict([vetor]))[0]
    proba = modelo.predict_proba([vetor])[0]
    conf  = {c: round(p*100,1) for c,p in zip(le_target.classes_, proba)}
    return pred, conf

def gerar_plano_acao(env:int, soc:int, gov:int, importancias:list) -> List[ItemPlanoAcao]:
    imp = dict(importancias)
    pilares = {
        'E': (env, imp.get('environment_score', 0.33)),
        'S': (soc, imp.get('social_score',      0.33)),
        'G': (gov, imp.get('governance_score',  0.33)),
    }
    return [
        ItemPlanoAcao(pilar=p, score=score, importancia=imp_v, acao=acao)
        for p,(score,imp_v) in sorted(pilares.items(), key=lambda x: -x[1][1])
        if score < LIMIAR_CONFORMIDADE
        for acao in PLANOS_ACAO[p][:3]
    ]

def diagnosticar(empresa: Empresa, pesos: PesosSetor, benchmark: dict,
                 knn_pack: dict, rf_pack: dict, config: dict) -> DiagnosticoESG:
    le_industry = config['le_industry']
    env = empresa.scores.environment_score
    soc = empresa.scores.social_score
    gov = empresa.scores.governance_score
    ind_enc   = _encode_industria(empresa.industry, le_industry)
    vetor     = [env, soc, gov, ind_enc]
    total     = empresa.scores.total
    sp        = calcular_score_ponderado(env, soc, gov, pesos)
    risco     = calcular_risco(sp)
    impacto   = calcular_impacto_empresa(sp, empresa.industry, benchmark)
    quadrante = Quadrante.classificar(impacto, risco)
    level, grade = mapear_grade(total)
    pred_knn, conf_knn, _ = classificar_knn(vetor, knn_pack)
    pred_rf,  conf_rf     = classificar_rf(vetor,  rf_pack)
    plano = gerar_plano_acao(env, soc, gov, rf_pack['importancias'])
    return DiagnosticoESG(
        name=empresa.name, industry=empresa.industry,
        environment_score=env, social_score=soc, governance_score=gov,
        total_score=total, score_ponderado=sp,
        w_E=pesos.w_E, w_S=pesos.w_S, w_G=pesos.w_G, fonte_pesos=pesos.fonte,
        grade=grade, level=level, risco=risco, impacto=impacto, quadrante=quadrante,
        maturidade_rf=Maturidade.de_nivel(pred_rf),
        confianca_rf_high=conf_rf.get('High',0), confianca_rf_medium=conf_rf.get('Medium',0),
        maturidade_knn=Maturidade.de_nivel(pred_knn),
        confianca_knn_high=conf_knn.get('High',0), confianca_knn_medium=conf_knn.get('Medium',0),
        plano_acao=plano)
