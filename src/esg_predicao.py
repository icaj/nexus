import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

PASTA_SAIDA = "saida_esg"

PLANOS_ACAO = {
    'E': [
        "Implementar processo formal de gestão de impactos ambientais (PGRSS/PGRS).",
        "Realizar levantamento e estimativa da pegada de carbono (GHG Protocol).",
        "Buscar certificação ISO 14001 ou equivalente de gestão ambiental.",
        "Implantar programa de eficiência energética e gestão de resíduos.",
        "Contratar auditoria externa do inventário de Gases de Efeito Estufa.",
    ],
    'S': [
        "Formalizar política de compromisso com trabalho digno e direitos humanos.",
        "Implementar programa estruturado de diversidade, equidade e inclusão.",
        "Criar programa de saúde mental e bem-estar para colaboradores.",
        "Desenvolver ações de engajamento e voluntariado com a comunidade.",
        "Realizar treinamentos regulares de sustentabilidade para colaboradores.",
    ],
    'G': [
        "Aprovar e publicar política formal de responsabilidade socioambiental.",
        "Implantar código de conduta anticorrupção e canal de denúncias.",
        "Incluir cláusulas ESG nos contratos com fornecedores.",
        "Definir critérios socioambientais para seleção de fornecedores.",
        "Realizar auditorias ESG periódicas na cadeia de suprimentos.",
    ],
}

GRADE_MAP = {
    range(0,    600):  ('Abaixo do mínimo', 'D'),
    range(600,  750):  ('Baixo',            'B'),
    range(750,  900):  ('Baixo-Médio',      'BB'),
    range(900,  1200): ('Médio-Alto',       'BBB'),
    range(1200, 1800): ('Alto',             'A'),
    range(1800, 3001): ('Excelente',        'AA+'),
}

def mapear_grade(total_score: int) -> tuple:
    for r, (level, grade) in GRADE_MAP.items():
        if total_score in r:
            return level, grade
    return ('Desconhecido', '?')


def carregar_modelos():
    try:
        knn_pack = joblib.load(f"{PASTA_SAIDA}/modelo_knn.pkl")
        rf_pack  = joblib.load(f"{PASTA_SAIDA}/modelo_rf.pkl")
        config   = joblib.load(f"{PASTA_SAIDA}/config.pkl")
        return knn_pack, rf_pack, config
    except FileNotFoundError as e:
        print(f"\n[ERRO] {e}")
        print("Execute primeiro: python esg_pipeline.py")
        sys.exit(1)


def obter_pesos(industry: str, config: dict) -> tuple:
    pesos_por_ind = config.get('pesos_por_ind', {})
    pesos_global  = config.get('pesos_global', {
        'environment_score': 0.3865,
        'social_score':      0.3267,
        'governance_score':  0.2869,
    })
    p = pesos_por_ind.get(industry)
    if p:
        return p['w_E'], p['w_S'], p['w_G'], p['fonte']
    else:
        return (pesos_global['environment_score'],
                pesos_global['social_score'],
                pesos_global['governance_score'],
                'fallback (setor não encontrado na base)')


def calcular_score_ponderado(env: int, soc: int, gov: int, w_e: float, w_s: float, w_g: float) -> float:
    """
    Score ponderado = média ponderada dos três pilares com pesos do setor.
    Fórmula: score_pond = w_E × env + w_S × soc + w_G × gov
    Resultado na escala 0–1000.
    """
    return round(w_e * env + w_s * soc + w_g * gov, 1)


def calcular_risco(score_ponderado: float) -> float:
    """
    Risco ESG = gap percentual até o máximo (1000) do score ponderado.
    Fórmula: (1000 − score_ponderado) / 1000 × 100
    Os pesos do setor fazem o risco refletir o que mais importa
    para aquela indústria — não apenas a média simples dos pilares.
    """
    return round((1000 - score_ponderado) / 1000 * 100, 1)


def calcular_impacto(score_ponderado: float, industry: str, config: dict) -> float:
    """
    Impacto = posição relativa do score ponderado da empresa
    em relação à média ponderada do setor na base de treino.
    Resultado 0–100: > 50 significa acima da média do setor.
    """
    bench = config.get('benchmark', {}).get('score_ponderado', {})
    media_setor  = bench.get(industry, 355.0)
    desvio_aprox = 110.0
    z = (score_ponderado - media_setor) / desvio_aprox
    return round(max(0.0, min(100.0, 50 + z * 25)), 1)


def classificar_quadrante(impacto: float, risco: float) -> str:
    hi = impacto > 50
    hr = risco   > 50
    if   hi and hr:      return "Alto Impacto / Alto Risco"
    elif hi and not hr:  return "Alto Impacto / Baixo Risco"
    elif not hi and hr:  return "Baixo Impacto / Alto Risco"
    else:                return "Baixo Impacto / Baixo Risco"


def gerar_plano(env_score, soc_score, gov_score, importancias_rf):
    """
    Identifica pilares com score abaixo de 400 (abaixo de BBB)
    e os ordena pela importância aprendida pelo Random Forest.
    """
    LIMIAR = 400
    imp = dict(importancias_rf)
    pilares = {
        'E': (env_score, imp.get('environment_score', 0.33)),
        'S': (soc_score, imp.get('social_score',      0.33)),
        'G': (gov_score, imp.get('governance_score',  0.33)),
    }
    # Ordenar por importância (mais crítico primeiro)
    ordenados = sorted(pilares.items(), key=lambda x: -x[1][1])

    plano = []
    for pilar, (score, importancia) in ordenados:
        if score < LIMIAR:
            for acao in PLANOS_ACAO[pilar][:3]:
                plano.append({'pilar': pilar, 'score': score,
                              'importancia': importancia, 'acao': acao})
    return plano


def classificar_empresa(nome, industry, env_score, soc_score, gov_score, knn_pack, rf_pack, config):
    """Pipeline completo para uma empresa."""

    le_ind    = config['le_industry']
    le_target = knn_pack['le_target']
    features  = knn_pack['features']

    # Encoding da indústria
    industry_orig = industry
    try:
        ind_enc = le_ind.transform([industry])[0]
    except ValueError:
        ind_enc = le_ind.transform(['Unknown'])[0]
        industry = 'Unknown'

    total_score = env_score + soc_score + gov_score
    vetor = [env_score, soc_score, gov_score, ind_enc]

    # Pesos do setor
    w_e, w_s, w_g, fonte_peso = obter_pesos(industry_orig, config)
    score_pond = calcular_score_ponderado(env_score, soc_score, gov_score, w_e, w_s, w_g)

    # KNN
    knn       = knn_pack['modelo']
    pred_knn  = le_target.inverse_transform(knn.predict([vetor]))[0]
    proba_knn = knn.predict_proba([vetor])[0]
    conf_knn  = {c: round(p*100,1) for c,p in zip(le_target.classes_, proba_knn)}

    # Vizinhos mais próximos (benchmarking)
    dist, idx = knn.named_steps['knn'].kneighbors(
        knn.named_steps['scaler'].transform([vetor]),
        n_neighbors=3
    )
    base_ref = knn_pack['base_ref']
    vizinhos = []
    for d, i in zip(dist[0], idx[0]):
        emp = base_ref[i]
        vizinhos.append({
            'nome':           emp['name'],
            'industry':       emp['industry'],
            'maturidade':     emp['maturidade'],
            'total_score':    emp['total_score'],
            'score_pond':     emp.get('score_ponderado', '—'),
            'distancia':      round(d, 2),
        })

    # Random Forest
    rf       = rf_pack['modelo']
    pred_rf  = le_target.inverse_transform(rf.predict([vetor]))[0]
    proba_rf = rf.predict_proba([vetor])[0]
    conf_rf  = {c: round(p*100,1) for c,p in zip(le_target.classes_, proba_rf)}

    # Métricas ponderadas
    risco     = calcular_risco(score_pond)
    impacto   = calcular_impacto(score_pond, industry_orig, config)
    quadrante = classificar_quadrante(impacto, risco)
    level, grade = mapear_grade(total_score)
    importancias_rf = rf_pack['importancias']
    plano     = gerar_plano(env_score, soc_score, gov_score, importancias_rf)

    # Relatório
    sep  = "═" * 62
    sep2 = "─" * 62
    mapa = config['mapa_maturidade']

    print(f"\n{sep}")
    print(f"  DIAGNÓSTICO ESG — {nome}")
    print(f"  Setor: {industry_orig}")
    print(f"{sep}")

    print(f"\n  SCORES (escala ESG Enterprise, 0–1000 por pilar)")
    print(f"  {'Pilar':<22} {'Score':>8}  {'Peso setor':>10}  {'Grade ref.'}")
    print(f"  {'─'*52}")
    def grade_pilar(s):
        if s >= 600: return 'AA'
        elif s >= 500: return 'A'
        elif s >= 400: return 'BBB'
        elif s >= 300: return 'BB'
        elif s >= 200: return 'B'
        else: return '<B'

    print(f"  {'Ambiental (E)':<22} {env_score:>8}  {w_e:>10.3f}  {grade_pilar(env_score)}")
    print(f"  {'Social (S)':<22} {soc_score:>8}  {w_s:>10.3f}  {grade_pilar(soc_score)}")
    print(f"  {'Governança (G)':<22} {gov_score:>8}  {w_g:>10.3f}  {grade_pilar(gov_score)}")
    print(f"  {'─'*52}")
    print(f"  {'Score ponderado':<22} {score_pond:>8.1f}  {'(w_E+w_S+w_G=1)':>10}  {grade}")
    print(f"  {'Total (E+S+G)':<22} {total_score:>8}  {'':>10}  {grade}  ({level})")
    print(f"\n  Origem dos pesos: {fonte_peso}")

    print(f"\n  RISCO ESG (ponderado por setor)")
    print(f"  Fórmula  : (1000 − score_ponderado) / 1000 × 100")
    print(f"  Cálculo  : (1000 − {score_pond}) / 1000 × 100")
    print(f"  Risco    : {risco}%  ← gap de não-conformidade ESG")

    print(f"\n  IMPACTO (ponderado por setor)")
    print(f"  Score ponderado ({score_pond}) vs média do setor '{industry_orig}'")
    print(f"  Impacto  : {impacto} / 100")

    print(f"\n{sep2}")
    print(f"  QUADRANTE: {quadrante}")

    print(f"\n{sep2}")
    print(f"  [KNN] MATURIDADE:  {pred_knn}  ({mapa.get(pred_knn, pred_knn)})")
    print(f"  Confiança:")
    for cls, pct in sorted(conf_knn.items(), key=lambda x:-x[1]):
        print(f"    {cls:<8} {pct:5.1f}%  {'█'*int(pct/4)}")

    print(f"\n  [RANDOM FOREST] MATURIDADE:  {pred_rf}  ({mapa.get(pred_rf, pred_rf)})")
    print(f"  Confiança:")
    for cls, pct in sorted(conf_rf.items(), key=lambda x:-x[1]):
        print(f"    {cls:<8} {pct:5.1f}%  {'█'*int(pct/4)}")

    print(f"\n{sep2}")
    print(f"  BENCHMARKING — 3 empresas mais similares")
    for i, v in enumerate(vizinhos, 1):
        print(f"  {i}. {v['nome'][:45]:<45}")
        print(f"     {v['industry']} | {v['maturidade']} | Score: {v['total_score']} | Dist: {v['distancia']}")

    print(f"\n{sep2}")
    print(f"  PLANO DE AÇÃO (pilares com score < 400 / grade < BBB)")
    if not plano:
        print(f"\n  Todos os pilares em nível BBB ou superior.")
        print(f"  Recomendação: manter e evoluir para certificações externas.")
    else:
        pilar_atual = None
        nomes = {'E': 'AMBIENTAL', 'S': 'SOCIAL', 'G': 'GOVERNANÇA'}
        for item in plano:
            if item['pilar'] != pilar_atual:
                pilar_atual = item['pilar']
                print(f"\n  {nomes[pilar_atual]} (score={item['score']} | "
                      f"importância RF={item['importancia']:.3f}):")
            print(f"    • {item['acao']}")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────
# MODO 1 — ARQUIVO CSV
# ──────────────────────────────────────────────────────────────────

def processar_arquivo(caminho: str):
    knn_pack, rf_pack, config = carregar_modelos()
    try:
        df = pd.read_csv(caminho) if caminho.endswith('.csv') else pd.read_excel(caminho)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {caminho}"); sys.exit(1)

    print(f"\n{len(df)} empresa(s) encontrada(s) em '{caminho}'")

    colunas_req = ['name','industry','environment_score','social_score','governance_score']
    for col in colunas_req:
        if col not in df.columns:
            print(f"[ERRO] Coluna obrigatória ausente: {col}")
            print(f"Colunas esperadas: {colunas_req}")
            sys.exit(1)

    for _, row in df.iterrows():
        classificar_empresa(
            nome=str(row['name']),
            industry=str(row.get('industry','Unknown')),
            env_score=int(row['environment_score']),
            soc_score=int(row['social_score']),
            gov_score=int(row['governance_score']),
            knn_pack=knn_pack, rf_pack=rf_pack, config=config
        )


# ──────────────────────────────────────────────────────────────────
# MODO 2 — TERMINAL INTERATIVO
# ──────────────────────────────────────────────────────────────────

def entrada_interativa():
    knn_pack, rf_pack, config = carregar_modelos()

    print("\n" + "═"*62)
    print("  DIAGNÓSTICO ESG — Entrada Interativa")
    print("  Insira os scores na escala ESG Enterprise (0–1000)")
    print("═"*62)

    nome = input("\nNome da empresa: ").strip() or "Nova Empresa"
    industry = input("Setor (ex: Technology, Energy, Banking): ").strip() or "Unknown"

    def ler_score(label):
        while True:
            try:
                v = int(input(f"Score {label} (0–1000): ").strip())
                if 0 <= v <= 1000:
                    return v
                print("  Valor deve ser entre 0 e 1000.")
            except ValueError:
                print("  Digite um número inteiro.")

    env = ler_score("Ambiental")
    soc = ler_score("Social")
    gov = ler_score("Governança")

    classificar_empresa(nome, industry, env, soc, gov, knn_pack, rf_pack, config)

# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classificação ESG — Edenred Brasil")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--arquivo", "-a", metavar="ARQUIVO", help="CSV/Excel com colunas: name, industry, environment_score, social_score, governance_score")
    grupo.add_argument("--interativo", "-i", action="store_true", help="Entrada manual no terminal")
    args = parser.parse_args()

    if args.interativo:
        entrada_interativa()
    else:
        processar_arquivo(args.arquivo)
