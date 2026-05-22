from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import os
import kagglehub
import shutil

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────
# ETAPA 0 — ENTENDIMENTO DOS DADOS E METODOLOGIA
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────

CAMINHO_DADOS  = "./data/raw/data.csv"
PASTA_BRONZE   = "./data/bronze"
PASTA_SAIDA    = "./data/processado"
PASTA_MODELOS  = "./modelos"

os.makedirs(PASTA_BRONZE,  exist_ok=True)
os.makedirs(PASTA_SAIDA,   exist_ok=True)
os.makedirs(PASTA_MODELOS, exist_ok=True)

# Colunas que não entram nos modelos
COLUNAS_REMOVER = ['ticker', 'logo', 'weburl', 'last_processing_date', 'cik', 'currency', 'exchange']

# Features numéricas dos scores dos pilares
FEATURES_SCORES = ['environment_score', 'social_score', 'governance_score']

# Feature categórica
FEATURE_INDUSTRY = 'industry'

# Variável-alvo
TARGET = 'total_level'   # 'High' ou 'Medium'

# Mapa de maturidade (total_level → rótulo interpretativo)
MAPA_MATURIDADE = {'High': 'Avançado', 'Medium': 'Iniciante'}

# Número mínimo de empresas por setor para calcular pesos empíricos.
# Setores abaixo desse limiar recebem os pesos globais (fallback).
MIN_EMPRESAS_PESO = 5

def carregar_dados_kaggle():
    # obtém base de dados do KaggleHub (dataset público de ratings ESG de empresas)
    path = kagglehub.dataset_download("alistairking/public-company-esg-ratings-dataset")
    arquivos = os.listdir(path)
    arquivo_csv = os.path.join(path, arquivos[0])
    shutil.copy(arquivo_csv, CAMINHO_DADOS)

# ──────────────────────────────────────────────────────────────────
# ETAPA 1 — PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────────────────────────

def etapa1_preprocessamento(caminho: str): 
    print("\n" + "═"*62)
    print("ETAPA 1 — PRÉ-PROCESSAMENTO")
    print("═"*62)

    df = pd.read_csv(caminho)
    print(f"\nBase carregada: {df.shape[0]} empresas × {df.shape[1]} colunas")

    # Remover colunas não analíticas
    df = df.drop(columns=[c for c in COLUNAS_REMOVER if c in df.columns])
    print(f"Colunas após remoção de metadados: {df.shape[1]}")

    # Tratar nulos de industry
    n_nulos = df['industry'].isna().sum()
    df['industry'] = df['industry'].fillna('Unknown')
    print(f"Nulos em 'industry': {n_nulos} → preenchidos com 'Unknown'")

    # Normalizar nomes de indústria
    # Problema: o mesmo setor aparece com variações de nome na base
    # (uso de '&' vs 'and', vírgulas, espaços extras).
    # Solução: mapeamento explícito para um nome canônico único.
    # Regra geral adotada: usar "and" no lugar de "&" e sem vírgulas.
    NOMES_CANONICOS = {
        'Aerospace & Defense':           'Aerospace and Defense',
        'Hotels, Restaurants & Leisure': 'Hotels Restaurants and Leisure',
        'Metals & Mining':               'Metals and Mining',
    }

    df['industry'] = df['industry'].str.strip()

    antes = df['industry'].value_counts().to_dict()
    df['industry'] = df['industry'].replace(NOMES_CANONICOS)
    depois = df['industry'].value_counts().to_dict()

    print(f"\nNormalização de nomes de indústria:")
    for original, canonico in NOMES_CANONICOS.items():
        n_antes  = antes.get(original, 0)
        n_depois = depois.get(canonico, 0)
        print(f"  '{original}'")
        print(f"    → '{canonico}'  ({n_antes} registros fundidos, total no grupo: {n_depois})")
    print(f"\n  Setores únicos antes : {len(antes)}")
    print(f"  Setores únicos depois: {df['industry'].nunique()}")

    # Verificar: total_score == soma dos pilares
    soma = df[FEATURES_SCORES].sum(axis=1)
    assert (soma == df['total_score']).all(), "ERRO: total_score != soma dos pilares"
    print("Verificação: total_score = soma(E+S+G) ✓")

    # Verificar cobertura de valores
    print(f"\nDistribuição do alvo ({TARGET}):")
    for nivel, n in df[TARGET].value_counts().items():
        print(f"  {nivel:<10}: {n:3d}  ({n/len(df)*100:.1f}%)")

    print(f"\nGrades totais disponíveis na base:")
    grade_info = df.groupby('total_grade')['total_score'].agg(['min','max','count'])
    print(grade_info.sort_values('min').to_string())

    return df


# ──────────────────────────────────────────────────────────────────
# ETAPA 2 — CÁLCULO DE MATURIDADE, RISCO E IMPACTO
# ──────────────────────────────────────────────────────────────────

def calcular_pesos_por_industria(df: pd.DataFrame) -> dict:
    # Pesos globais (fallback) — correlação na base inteira
    corr_global = (df[FEATURES_SCORES]
                   .corrwith(df['total_score'])
                   .clip(lower=0))
    soma_global = corr_global.sum()
    pesos_global = (corr_global / soma_global).to_dict()

    pesos = {}
    for ind, grupo in df.groupby('industry'):
        n = len(grupo)
        if n >= MIN_EMPRESAS_PESO:
            corr = (grupo[FEATURES_SCORES]
                    .corrwith(grupo['total_score'])
                    .clip(lower=0))
            soma = corr.sum()
            if soma == 0:
                w = pesos_global
                fonte = 'fallback (corr=0)'
            else:
                w = (corr / soma).to_dict()
                fonte = f'empírico (n={n})'
        else:
            w = pesos_global
            fonte = f'fallback (n={n}<{MIN_EMPRESAS_PESO})'

        pesos[ind] = {
            'w_E':   round(w['environment_score'], 4),
            'w_S':   round(w['social_score'],      4),
            'w_G':   round(w['governance_score'],  4),
            'n':     n,
            'fonte': fonte,
        }

    return pesos, pesos_global


def etapa2_metricas(df: pd.DataFrame) -> tuple:
    print("\n" + "═"*62)
    print("ETAPA 2 — CÁLCULO DE MATURIDADE, RISCO E IMPACTO")
    print("═"*62)

    # ── Pesos por indústria ──────────────────────────────────────
    pesos_por_ind, pesos_global = calcular_pesos_por_industria(df)

    print(f"\nPesos globais (fallback para n < {MIN_EMPRESAS_PESO}):")
    print(f"  w_E = {pesos_global['environment_score']:.4f}")
    print(f"  w_S = {pesos_global['social_score']:.4f}")
    print(f"  w_G = {pesos_global['governance_score']:.4f}")

    n_empirico = sum(1 for v in pesos_por_ind.values() if 'empírico' in v['fonte'])
    n_fallback = sum(1 for v in pesos_por_ind.values() if 'fallback' in v['fonte'])
    print(f"\nSetores com pesos empíricos : {n_empirico}")
    print(f"Setores com fallback        : {n_fallback}")

    # Mostrar exemplos de setores com pesos distintos
    print("\nAmostra de pesos por setor (variação mais expressiva):")
    pesos_df = pd.DataFrame(pesos_por_ind).T[['w_E','w_S','w_G','n','fonte']]
    # Ordenar pelo maior desvio de w_E em relação à global
    w_E_global = pesos_global['environment_score']
    pesos_df['desvio_E'] = (pesos_df['w_E'].astype(float) - w_E_global).abs()
    print(pesos_df.sort_values('desvio_E', ascending=False).head(8)[
        ['w_E','w_S','w_G','n','fonte']
    ].to_string())

    # ── Score ponderado por empresa ──────────────────────────────
    def score_ponderado(row):
        p = pesos_por_ind.get(row['industry'], {})
        w_e = p.get('w_E', pesos_global['environment_score'])
        w_s = p.get('w_S', pesos_global['social_score'])
        w_g = p.get('w_G', pesos_global['governance_score'])
        return (w_e * row['environment_score'] +
                w_s * row['social_score'] +
                w_g * row['governance_score'])

    df['score_ponderado'] = df.apply(score_ponderado, axis=1).round(1)

    # ── Maturidade ───────────────────────────────────────────────
    df['maturidade'] = df[TARGET].map(MAPA_MATURIDADE)

    # ── Risco ponderado ──────────────────────────────────────────
    # score_ponderado está na escala 0–1000 (média ponderada dos pilares)
    # Risco = gap percentual até o máximo (1000)
    df['risco'] = ((1000 - df['score_ponderado']) / 1000 * 100).round(1)

    # ── Impacto ponderado ────────────────────────────────────────
    # Percentil do score_ponderado dentro da indústria
    df['impacto'] = (
        df.groupby('industry')['score_ponderado']
          .rank(pct=True)
          .mul(100)
          .round(1)
    )

    # ── Quadrante ────────────────────────────────────────────────
    def quadrante(row):
        hi = row['impacto'] > 50
        hr = row['risco']   > 50
        if   hi and hr:      return "Alto Impacto / Alto Risco"
        elif hi and not hr:  return "Alto Impacto / Baixo Risco"
        elif not hi and hr:  return "Baixo Impacto / Alto Risco"
        else:                 return "Baixo Impacto / Baixo Risco"

    df['quadrante'] = df.apply(quadrante, axis=1)

    print("\nScore ponderado vs total_score (verificação):")
    print(df[['total_score','score_ponderado','risco','impacto']].describe().round(1))

    print("\nRisco médio por grade (usando score ponderado):")
    print(df.groupby('total_grade')['risco'].agg(['mean','min','max']).round(1))

    print("\nDistribuição por quadrante:")
    for q, n in df['quadrante'].value_counts().items():
        print(f"  {q:<38}: {n}")

    return df, pesos_por_ind, pesos_global


# ──────────────────────────────────────────────────────────────────
# ETAPA 3 — ANÁLISE EXPLORATÓRIA
# ──────────────────────────────────────────────────────────────────

def etapa3_exploracao(df: pd.DataFrame):
    print("\n" + "═"*62)
    print("ETAPA 3 — ANÁLISE EXPLORATÓRIA")
    print("═"*62)

    fig = plt.figure(figsize=(18, 14), facecolor='#F7F8FA')
    fig.suptitle('Análise ESG — Base de 722 Empresas (ESG Enterprise)',
                 fontsize=16, fontweight='bold', color='#1A1A2E', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.05, left=0.07, right=0.97)

    CORES = {'High':'#2ECC71', 'Medium':'#E74C3C'}
    COLS  = ['#3498DB','#E74C3C','#F39C12']

    # 1. Distribuição total_level
    ax1 = fig.add_subplot(gs[0, 0])
    vc = df['total_level'].value_counts()
    wedges, texts, autotexts = ax1.pie(
        vc.values, labels=vc.index,
        colors=[CORES[k] for k in vc.index],
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
    )
    for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
    ax1.set_title('Distribuição de Maturidade', fontweight='bold', fontsize=11)

    # 2. Distribuição total_grade
    ax2 = fig.add_subplot(gs[0, 1])
    grades = ['B','BB','BBB','A']
    counts = [df['total_grade'].value_counts().get(g,0) for g in grades]
    bars = ax2.bar(grades, counts,
                   color=['#E74C3C','#E67E22','#3498DB','#2ECC71'],
                   edgecolor='white', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, counts):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 str(val), ha='center', fontsize=10, fontweight='bold')
    ax2.set_title('Empresas por Grade Total', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Nº de empresas'); ax2.set_facecolor('#FAFAFA')
    ax2.spines[['top','right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Distribuição dos scores
    ax3 = fig.add_subplot(gs[0, 2])
    for col, cor, label in zip(
        FEATURES_SCORES, COLS,
        ['Ambiental','Social','Governança']
    ):
        ax3.hist(df[col], bins=25, alpha=0.6, color=cor, label=label)
    ax3.set_title('Distribuição dos Scores por Pilar', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Score (0–1000)'); ax3.legend(fontsize=8)
    ax3.set_facecolor('#FAFAFA')
    ax3.spines[['top','right']].set_visible(False)

    # 4. Scatter E vs S colorido por maturidade
    ax4 = fig.add_subplot(gs[1, 0])
    for nivel, cor in CORES.items():
        sub = df[df['total_level']==nivel]
        ax4.scatter(sub['environment_score'], sub['social_score'],
                    c=cor, alpha=0.5, s=20, label=nivel)
    ax4.set_xlabel('Environment Score'); ax4.set_ylabel('Social Score')
    ax4.set_title('E × S por Maturidade', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=8); ax4.set_facecolor('#FAFAFA')
    ax4.spines[['top','right']].set_visible(False)

    # 5. Score médio por indústria (top 12)
    ax5 = fig.add_subplot(gs[1, 1:])
    top_ind = df.groupby('industry')['total_score'].mean().sort_values(ascending=False).head(12)
    colors_bar = ['#2ECC71' if v >= 900 else '#E74C3C' for v in top_ind.values]
    ax5.barh(range(len(top_ind)), top_ind.values, color=colors_bar,
             edgecolor='white', height=0.7)
    ax5.set_yticks(range(len(top_ind)))
    ax5.set_yticklabels(top_ind.index, fontsize=8)
    ax5.axvline(900, color='#2ECC71', ls='--', lw=1.5, alpha=0.7, label='Limiar High (900)')
    ax5.set_title('Score ESG Médio por Indústria (Top 12)', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Total Score'); ax5.legend(fontsize=8)
    ax5.set_facecolor('#FAFAFA'); ax5.grid(axis='x', alpha=0.3)
    ax5.spines[['top','right']].set_visible(False)

    # 6. Matriz de criticidade
    ax6 = fig.add_subplot(gs[2, :2])
    cores_quad = {
        'Alto Impacto / Alto Risco':    '#E74C3C',
        'Alto Impacto / Baixo Risco':   '#F1C40F',
        'Baixo Impacto / Alto Risco':   '#E67E22',
        'Baixo Impacto / Baixo Risco':  '#2ECC71',
    }
    for quad, cor in cores_quad.items():
        sub = df[df['quadrante']==quad]
        ax6.scatter(sub['impacto'], sub['risco'], c=cor, alpha=0.4, s=15, label=f"{quad} (n={len(sub)})")
    ax6.axvline(50, color='gray', ls='--', lw=1, alpha=0.5)
    ax6.axhline(50, color='gray', ls='--', lw=1, alpha=0.5)
    ax6.set_xlabel('Impacto (percentil environment_score no setor)')
    ax6.set_ylabel('Risco ESG (%)')
    ax6.set_title('Matriz de Criticidade', fontweight='bold', fontsize=11)
    ax6.legend(fontsize=7, loc='upper left'); ax6.set_facecolor('#FAFAFA')
    ax6.spines[['top','right']].set_visible(False)

    # 7. Risco por grade
    ax7 = fig.add_subplot(gs[2, 2])
    ordem = ['A','BBB','BB','B']
    data_box = [df[df['total_grade']==g]['risco'].values for g in ordem]
    bp = ax7.boxplot(data_box, labels=ordem, patch_artist=True)
    cores_box = ['#2ECC71','#3498DB','#E67E22','#E74C3C']
    for patch, cor in zip(bp['boxes'], cores_box):
        patch.set_facecolor(cor); patch.set_alpha(0.7)
    ax7.set_title('Distribuição do Risco por Grade', fontweight='bold', fontsize=11)
    ax7.set_xlabel('Grade'); ax7.set_ylabel('Risco ESG (%)')
    ax7.set_facecolor('#FAFAFA')
    ax7.spines[['top','right']].set_visible(False)

    plt.savefig(f'{PASTA_SAIDA}/analise_exploratoria.png',
                dpi=140, bbox_inches='tight', facecolor='#F7F8FA')
    plt.close()
    print(f"\nGráfico salvo: {PASTA_SAIDA}/analise_exploratoria.png")

    # Estatísticas textuais
    print("\nCorrelação entre features e total_score:")
    print(df[FEATURES_SCORES+['total_score']].corr()['total_score'].drop('total_score').round(3))

    print("\nScore médio por maturidade:")
    print(df.groupby('maturidade')[
        FEATURES_SCORES+['total_score','risco','impacto']
    ].mean().round(1))


# ──────────────────────────────────────────────────────────────────
# ETAPA 4 — PREPARAÇÃO DAS FEATURES PARA ML
# ──────────────────────────────────────────────────────────────────

def etapa4_preparar_features(df: pd.DataFrame):
    """
    Features de entrada dos modelos:
      - environment_score  (0–1000, numérica)
      - social_score       (0–1000, numérica)
      - governance_score   (0–1000, numérica)
      - industry           (categórica → encoding ordinal)

    Variável-alvo:
      - total_level: 'High' (Avançado) ou 'Medium' (Iniciante)

    Por que não usar total_score diretamente como feature?
    Porque total_score = E + S + G — incluir os três pilares E os scores
    individuais criaria multicolinearidade perfeita. Usamos os pilares.

    Por que incluir industry?
    Setor captura contexto de exposição ESG. Uma empresa de Biotechnology
    com score 800 está em posição diferente de uma Utility com score 800.
    """
    print("\n" + "═"*62)
    print("ETAPA 4 — PREPARAÇÃO DAS FEATURES")
    print("═"*62)

    # Encoding da indústria
    le_ind = LabelEncoder()
    df['industry_enc'] = le_ind.fit_transform(df['industry'])

    FEATURES = FEATURES_SCORES + ['industry_enc']

    X = df[FEATURES].values
    y = df[TARGET].values

    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)  # Medium=0, High=1 (alfabético)

    print(f"\nFeatures: {FEATURES}")
    print(f"Shape X: {X.shape}")
    print(f"Classes: {le_target.classes_}  (encodadas como {list(range(len(le_target.classes_)))})")
    print(f"\nDistribuição y:")
    for cls, enc in zip(le_target.classes_, range(len(le_target.classes_))):
        n = (y_enc == enc).sum()
        print(f"  {cls} ({enc}): {n} ({n/len(y_enc)*100:.1f}%)")

    # Split estratificado: 80% treino, 20% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    print(f"\nSplit: {len(X_train)} treino / {len(X_test)} teste (80/20, estratificado)")

    return X_train, X_test, y_train, y_test, FEATURES, le_ind, le_target, df


# ──────────────────────────────────────────────────────────────────
# ETAPA 5 — TREINO DOS MODELOS
# ──────────────────────────────────────────────────────────────────

def etapa5_treino(X_train, X_test, y_train, y_test, FEATURES, le_target):
    print("\n" + "═"*62)
    print("ETAPA 5 — TREINO: KNN E RANDOM FOREST")
    print("═"*62)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── KNN ────────────────────────────────────────────────────────
    # O KNN classifica cada empresa nova calculando a distância
    # euclidiana para todas as 722 empresas do treino e votando
    # entre as k mais próximas.
    #
    # Com scores na escala 0-1000 e industry_enc em escala diferente,
    # aplicamos StandardScaler para colocar tudo na mesma escala
    # antes de calcular distâncias — caso contrário, environment_score
    # (que vai até 719) dominaria industry_enc (que vai até ~46).

    print("\n[KNN] Grid search de k com validação cruzada 5-fold...")

    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(metric='euclidean'))
    ])

    param_grid_knn = {'knn__n_neighbors': list(range(3, 22, 2))}

    grid_knn = GridSearchCV(
        knn_pipe, param_grid_knn, cv=cv,
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_knn.fit(X_train, y_train)

    melhor_k = grid_knn.best_params_['knn__n_neighbors']
    print(f"\n  Resultados por k:")
    resultados = grid_knn.cv_results_
    for k, mean, std in zip(
        param_grid_knn['knn__n_neighbors'],
        resultados['mean_test_score'],
        resultados['std_test_score']
    ):
        barra = "█" * int(mean * 40)
        print(f"  k={k:02d}  {mean:.2%} ± {std:.2%}  {barra}")

    print(f"\n  → Melhor k = {melhor_k}  (acurácia CV = {grid_knn.best_score_:.2%})")

    knn_final = grid_knn.best_estimator_
    y_pred_knn = knn_final.predict(X_test)
    acc_knn_test = accuracy_score(y_test, y_pred_knn)
    print(f"  Acurácia no teste: {acc_knn_test:.2%}")
    print("\n  Relatório completo (teste):")
    print(classification_report(
        y_test, y_pred_knn,
        target_names=le_target.classes_
    ))

    # ── RANDOM FOREST ──────────────────────────────────────────────
    # O Random Forest treina múltiplas árvores de decisão em
    # subconjuntos aleatórios dos dados (bagging) e combina as
    # predições por votação. É mais robusto que uma única árvore.
    #
    # Vantagens sobre KNN aqui:
    #   1. Produz importância de variáveis (interpretabilidade)
    #   2. Não é afetado pela escala das features
    #   3. Lida melhor com features de escalas muito diferentes
    #
    # Por que Random Forest em vez de Árvore de Decisão simples?
    # Com 722 exemplos e features numéricas de alta correlação,
    # uma única árvore tende ao overfitting. O RF estabiliza isso.

    print("\n[RANDOM FOREST] Grid search com validação cruzada 5-fold...")

    rf_pipe = Pipeline([
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid_rf = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth':    [4, 6, 8, None],
        'rf__min_samples_leaf': [5, 10, 15],
    }

    grid_rf = GridSearchCV(
        rf_pipe, param_grid_rf, cv=cv,
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_rf.fit(X_train, y_train)

    print(f"\n  → Melhores parâmetros: {grid_rf.best_params_}")
    print(f"  → Acurácia CV:         {grid_rf.best_score_:.2%}")

    rf_final = grid_rf.best_estimator_
    y_pred_rf = rf_final.predict(X_test)
    acc_rf_test = accuracy_score(y_test, y_pred_rf)
    print(f"  Acurácia no teste: {acc_rf_test:.2%}")
    print("\n  Relatório completo (teste):")
    print(classification_report(
        y_test, y_pred_rf,
        target_names=le_target.classes_
    ))

    # Importância das variáveis (Random Forest)
    rf_estimator = rf_final.named_steps['rf']
    importancias = sorted(
        zip(FEATURES, rf_estimator.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\n  Importância das variáveis (Random Forest):")
    for feat, imp in importancias:
        barra = "█" * int(imp * 50)
        print(f"    {feat:<25} {imp:.4f}  {barra}")

    return knn_final, rf_final, y_pred_knn, y_pred_rf, importancias, melhor_k


# ──────────────────────────────────────────────────────────────────
# ETAPA 6 — AVALIAÇÃO E VISUALIZAÇÃO
# ──────────────────────────────────────────────────────────────────

def etapa6_avaliacao(knn_final, rf_final, X_test, y_test,
                     y_pred_knn, y_pred_rf, importancias, le_target, FEATURES):
    print("\n" + "═"*62)
    print("ETAPA 6 — AVALIAÇÃO E VISUALIZAÇÕES")
    print("═"*62)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#F7F8FA')
    fig.suptitle('Avaliação dos Modelos ML — ESG', fontsize=14,
                 fontweight='bold', color='#1A1A2E')

    # Matriz de confusão KNN
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    disp = ConfusionMatrixDisplay(cm_knn, display_labels=le_target.classes_)
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f'KNN — Matriz de Confusão\nAcurácia: {accuracy_score(y_test,y_pred_knn):.2%}',
                      fontweight='bold')

    # Matriz de confusão Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp2 = ConfusionMatrixDisplay(cm_rf, display_labels=le_target.classes_)
    disp2.plot(ax=axes[1], colorbar=False, cmap='Greens')
    axes[1].set_title(f'Random Forest — Matriz de Confusão\nAcurácia: {accuracy_score(y_test,y_pred_rf):.2%}',
                      fontweight='bold')

    # Importância das variáveis (RF)
    feats  = [f[0] for f in importancias]
    valores = [f[1] for f in importancias]
    colors_imp = ['#3498DB' if v < 0.3 else ('#E67E22' if v < 0.6 else '#E74C3C')
                  for v in valores]
    axes[2].barh(range(len(feats)), valores, color=colors_imp,
                 edgecolor='white', height=0.6)
    axes[2].set_yticks(range(len(feats)))
    axes[2].set_yticklabels(feats, fontsize=9)
    axes[2].set_title('Importância das Variáveis\n(Random Forest)',
                      fontweight='bold')
    axes[2].set_xlabel('Importância (Gini)')
    axes[2].set_facecolor('#FAFAFA')
    axes[2].spines[['top','right']].set_visible(False)
    for i, val in enumerate(valores):
        axes[2].text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/avaliacao_modelos.png',
                dpi=140, bbox_inches='tight', facecolor='#F7F8FA')
    plt.close()
    print(f"Gráfico salvo: {PASTA_SAIDA}/avaliacao_modelos.png")

    # Comparativo final
    print(f"\n{'─'*40}")
    print(f"  COMPARATIVO FINAL")
    print(f"{'─'*40}")
    print(f"  Modelo         Acurácia (teste)")
    print(f"  KNN            {accuracy_score(y_test,y_pred_knn):.2%}")
    print(f"  Random Forest  {accuracy_score(y_test,y_pred_rf):.2%}")


# ──────────────────────────────────────────────────────────────────
# ETAPA 7 — SALVAR MODELOS E METADADOS
# ──────────────────────────────────────────────────────────────────

def etapa7_salvar(df, knn_final, rf_final, le_ind, le_target,
                  importancias, FEATURES, pesos_por_ind, pesos_global):
    print("\n" + "═"*62)
    print("ETAPA 7 — SALVANDO MODELOS")
    print("═"*62)

    # Base de treino completa para benchmarking por setor
    base_ref = df[['name','industry','environment_score','social_score',
                   'governance_score','total_score','score_ponderado',
                   'total_grade','total_level','maturidade',
                   'risco','impacto','quadrante']].copy()

    # Benchmark médio por indústria
    benchmark = df.groupby('industry')[
        ['environment_score','social_score','governance_score',
         'total_score','score_ponderado']
    ].mean().round(1).to_dict()

    joblib.dump({
        'modelo':       knn_final,
        'le_target':    le_target,
        'le_industry':  le_ind,
        'features':     FEATURES,
        'base_ref':     base_ref.to_dict('records'),
    }, f'{PASTA_MODELOS}/modelo_knn.pkl')
    print(f"   {PASTA_MODELOS}/modelo_knn.pkl")

    joblib.dump({
        'modelo':       rf_final,
        'le_target':    le_target,
        'le_industry':  le_ind,
        'features':     FEATURES,
        'importancias': importancias,
        'base_ref':     base_ref.to_dict('records'),
    }, f'{PASTA_MODELOS}/modelo_rf.pkl')
    print(f"   {PASTA_MODELOS}/modelo_rf.pkl")

    joblib.dump({
        'le_industry':      le_ind,
        'le_target':        le_target,
        'features':         FEATURES,
        'benchmark':        benchmark,
        'mapa_maturidade':  MAPA_MATURIDADE,
        'pesos_por_ind':    pesos_por_ind,
        'pesos_global':     pesos_global,
        'min_empresas_peso': MIN_EMPRESAS_PESO,
    }, f'{PASTA_MODELOS}/config.pkl')
    print(f"   {PASTA_MODELOS}/config.pkl")

    # Salvar tabela de pesos por indústria para documentação
    pd.DataFrame(pesos_por_ind).T[
        ['w_E','w_S','w_G','n','fonte']
    ].to_csv(f'{PASTA_BRONZE}/pesos_por_industria.csv')
    print(f"   {PASTA_BRONZE}/pesos_por_industria.csv")

    # Salvar base processada em CSV
    base_ref.to_csv(f'{PASTA_BRONZE}/base_processada.csv', index=False)
    print(f"   {PASTA_BRONZE}/base_processada.csv")

    print(f"\nPróximo passo: execute esg_predicao.py para classificar novos fornecedores.")

# ──────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PIPELINE ESG — BASE ESG ENTERPRISE                      ║")
    print("║  Edenred Brasil | CESAR School 2025                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    carregar_dados_kaggle()
    df = etapa1_preprocessamento(CAMINHO_DADOS)
    df, pesos_por_ind, pesos_global = etapa2_metricas(df)
    etapa3_exploracao(df)
    X_train, X_test, y_train, y_test, FEATURES, le_ind, le_target, df = etapa4_preparar_features(df)
    knn_final, rf_final, y_pred_knn, y_pred_rf, importancias, melhor_k = etapa5_treino(X_train, X_test, y_train, y_test,FEATURES, le_target)
    etapa6_avaliacao(knn_final, rf_final, X_test, y_test, y_pred_knn, y_pred_rf, importancias, le_target, FEATURES)
    etapa7_salvar(df, knn_final, rf_final, le_ind, le_target, importancias, FEATURES, pesos_por_ind, pesos_global)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  PIPELINE CONCLUÍDO                                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
