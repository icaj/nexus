from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from cliente_api import ClienteApiESG, ErroAPI, URL_API_PADRAO


st.set_page_config(
    page_title="ESG Nexus Enterprise",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(40, 45, 55, 0.12);
        border-radius: 14px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 22px rgba(20, 24, 32, 0.05);
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(40, 45, 55, 0.10);
    }
    .painel-info {
        padding: 16px 18px;
        border-radius: 12px;
        background: #f7f9fc;
        border: 1px solid rgba(40, 45, 55, 0.10);
        margin-bottom: 16px;
        color: #2b3440;
    }
    .titulo-produto {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitulo-produto {
        font-size: 0.9rem;
        color: #657184;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


COLUNAS_OBRIGATORIAS_IMPORTACAO = [
    "codigo_fornecedor",
    "razao_social",
    "cnpj",
    "setor",
    "pais",
    "possui_politica_ambiental",
    "emissoes_carbono_ton",
    "percentual_energia_renovavel",
    "percentual_reciclagem_residuos",
    "incidentes_trabalhistas_12m",
    "possui_programa_diversidade",
    "possui_politica_privacidade_dados",
    "possui_politica_anticorrupcao",
    "consta_lista_sancoes",
    "noticias_negativas_12m",
    "quantidade_certificacoes",
    "receita_anual",
]

COLUNAS_BOOLEANAS_IMPORTACAO = [
    "possui_politica_ambiental",
    "possui_programa_diversidade",
    "possui_politica_privacidade_dados",
    "possui_politica_anticorrupcao",
    "consta_lista_sancoes",
]

COLUNAS_NUMERICAS_IMPORTACAO = [
    "emissoes_carbono_ton",
    "percentual_energia_renovavel",
    "percentual_reciclagem_residuos",
    "incidentes_trabalhistas_12m",
    "noticias_negativas_12m",
    "quantidade_certificacoes",
    "receita_anual",
]


def normalizar_coluna(nome: str) -> str:
    return str(nome).strip().lower().replace(" ", "_").replace("-", "_")


def converter_bool(valor: object) -> bool:
    if isinstance(valor, bool):
        return valor
    if pd.isna(valor):
        return False
    return str(valor).strip().lower() in {"1", "true", "sim", "s", "yes", "y"}


def carregar_planilha_fornecedores(arquivo) -> pd.DataFrame:
    nome = arquivo.name.lower()
    if nome.endswith(".csv"):
        df = pd.read_csv(arquivo)
    else:
        df = pd.read_excel(arquivo)
    df = df.rename(columns={coluna: normalizar_coluna(coluna) for coluna in df.columns})
    return df


def validar_e_normalizar_importacao(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    erros: list[str] = []
    ausentes = [coluna for coluna in COLUNAS_OBRIGATORIAS_IMPORTACAO if coluna not in df.columns]
    if ausentes:
        erros.append("Colunas obrigatórias ausentes: " + ", ".join(ausentes))
        return df, erros

    normalizado = df[COLUNAS_OBRIGATORIAS_IMPORTACAO].copy()
    for coluna in ["codigo_fornecedor", "razao_social", "cnpj", "setor", "pais"]:
        normalizado[coluna] = normalizado[coluna].fillna("").astype(str).str.strip()
    for coluna in COLUNAS_BOOLEANAS_IMPORTACAO:
        normalizado[coluna] = normalizado[coluna].map(converter_bool)
    for coluna in COLUNAS_NUMERICAS_IMPORTACAO:
        normalizado[coluna] = pd.to_numeric(normalizado[coluna], errors="coerce").fillna(0)

    for indice, linha in normalizado.iterrows():
        numero_linha = indice + 2
        if not linha["codigo_fornecedor"]:
            erros.append(f"Linha {numero_linha}: código do fornecedor vazio.")
        if not linha["razao_social"]:
            erros.append(f"Linha {numero_linha}: razão social vazia.")
        if not linha["setor"]:
            erros.append(f"Linha {numero_linha}: setor vazio.")
        for coluna in ["percentual_energia_renovavel", "percentual_reciclagem_residuos"]:
            if float(linha[coluna]) < 0 or float(linha[coluna]) > 100:
                erros.append(f"Linha {numero_linha}: {coluna} deve estar entre 0 e 100.")
        for coluna in ["emissoes_carbono_ton", "incidentes_trabalhistas_12m", "noticias_negativas_12m", "quantidade_certificacoes", "receita_anual"]:
            if float(linha[coluna]) < 0:
                erros.append(f"Linha {numero_linha}: {coluna} não pode ser negativo.")

    duplicados = normalizado[normalizado["codigo_fornecedor"].duplicated()]["codigo_fornecedor"].tolist()
    if duplicados:
        erros.append("Códigos duplicados na planilha: " + ", ".join(sorted(set(duplicados))))
    return normalizado, erros


def converter_para_json_nativo(valor):
    if pd.isna(valor):
        return None
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def dataframe_para_registros_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []
    for registro in df.to_dict("records"):
        registros.append({campo: converter_para_json_nativo(valor) for campo, valor in registro.items()})
    return registros


def gerar_modelo_planilha() -> bytes:
    exemplo = pd.DataFrame([
        {
            "codigo_fornecedor": "FORN-001",
            "razao_social": "Fornecedor Exemplo LTDA",
            "cnpj": "00.000.000/0001-00",
            "setor": "Serviços",
            "pais": "BR",
            "possui_politica_ambiental": "sim",
            "emissoes_carbono_ton": 120,
            "percentual_energia_renovavel": 65,
            "percentual_reciclagem_residuos": 72,
            "incidentes_trabalhistas_12m": 0,
            "possui_programa_diversidade": "sim",
            "possui_politica_privacidade_dados": "sim",
            "possui_politica_anticorrupcao": "sim",
            "consta_lista_sancoes": "nao",
            "noticias_negativas_12m": 0,
            "quantidade_certificacoes": 2,
            "receita_anual": 1500000,
        }
    ])
    saida = io.BytesIO()
    with pd.ExcelWriter(saida, engine="openpyxl") as writer:
        exemplo.to_excel(writer, index=False, sheet_name="fornecedores")
    return saida.getvalue()


def gerar_excel_resultado(df_resultado: pd.DataFrame, df_erros: pd.DataFrame | None = None) -> bytes:
    saida = io.BytesIO()
    with pd.ExcelWriter(saida, engine="openpyxl") as writer:
        df_resultado.to_excel(writer, index=False, sheet_name="classificacao_esg")
        if df_erros is not None and not df_erros.empty:
            df_erros.to_excel(writer, index=False, sheet_name="erros")
    return saida.getvalue()


def paginas_permitidas(perfil: str) -> list[str]:
    mapa = {
        "administrador":   ["Dashboard Executivo ESG", "Dashboard Estatístico", "Importação e Classificação em Lote", "Fornecedores", "Classificação e Explicabilidade", "Dashboard de Machine Learning", "Operação ML/Airflow/MLflow"],
        "gerente_esg":     ["Dashboard Executivo ESG", "Dashboard Estatístico", "Importação e Classificação em Lote", "Fornecedores", "Classificação e Explicabilidade"],
        "analista_esg":    ["Dashboard Executivo ESG", "Dashboard Estatístico", "Importação e Classificação em Lote", "Fornecedores", "Classificação e Explicabilidade"],
        "operador_esg":    ["Importação e Classificação em Lote", "Fornecedores", "Classificação e Explicabilidade"],
        "cientista_dados": ["Dashboard Estatístico", "Dashboard de Machine Learning", "Classificação e Explicabilidade", "Operação ML/Airflow/MLflow"],
    }
    return mapa.get(perfil, ["Dashboard Executivo ESG"])


def obter_cliente() -> ClienteApiESG:
    return ClienteApiESG(
        url_base=st.session_state.get("url_api", URL_API_PADRAO),
        token=st.session_state.get("token"),
    )


def dados_executivos() -> dict[str, Any]:
    return obter_cliente().dashboard_executivo()


def dados_ml() -> dict[str, Any]:
    return obter_cliente().dashboard_ml()


def formulario_fornecedor(prefixo: str = "") -> dict[str, Any]:
    col1, col2, col3 = st.columns(3)
    with col1:
        codigo = st.text_input("Código", value="", key=f"{prefixo}codigo")
        razao = st.text_input("Razão social", value="", key=f"{prefixo}razao")
        cnpj = st.text_input("CNPJ", value="00.000.000/0001-00", key=f"{prefixo}cnpj")
        setor = st.selectbox("Setor", ["Serviços", "Indústria", "Tecnologia", "Logística", "Alimentos", "Energia"], key=f"{prefixo}setor")
        pais = st.text_input("País", value="BR", key=f"{prefixo}pais")
    with col2:
        politica_ambiental = st.checkbox("Possui política ambiental", value=True, key=f"{prefixo}ambiental")
        emissoes = st.number_input("Emissões de carbono (ton/ano)", min_value=0.0, value=120.0, step=10.0, key=f"{prefixo}emissoes")
        energia = st.slider("% energia renovável", 0.0, 100.0, 62.0, key=f"{prefixo}energia")
        reciclagem = st.slider("% reciclagem de resíduos", 0.0, 100.0, 70.0, key=f"{prefixo}reciclagem")
        certificacoes = st.number_input("Quantidade de certificações", min_value=0, value=2, step=1, key=f"{prefixo}certificacoes")
    with col3:
        incidentes = st.number_input("Incidentes trabalhistas 12m", min_value=0, value=0, step=1, key=f"{prefixo}incidentes")
        diversidade = st.checkbox("Programa de diversidade", value=True, key=f"{prefixo}diversidade")
        privacidade = st.checkbox("Política de privacidade", value=True, key=f"{prefixo}privacidade")
        anticorrupcao = st.checkbox("Política anticorrupção", value=True, key=f"{prefixo}anticorrupcao")
        sancoes = st.checkbox("Consta em lista de sanções", value=False, key=f"{prefixo}sancoes")
        noticias = st.number_input("Notícias negativas 12m", min_value=0, value=0, step=1, key=f"{prefixo}noticias")
        receita = st.number_input("Receita anual", min_value=0.0, value=1_000_000.0, step=50_000.0, key=f"{prefixo}receita")
    return {
        "codigo_fornecedor": codigo,
        "razao_social": razao,
        "cnpj": cnpj,
        "setor": setor,
        "pais": pais,
        "possui_politica_ambiental": politica_ambiental,
        "emissoes_carbono_ton": emissoes,
        "percentual_energia_renovavel": energia,
        "percentual_reciclagem_residuos": reciclagem,
        "incidentes_trabalhistas_12m": incidentes,
        "possui_programa_diversidade": diversidade,
        "possui_politica_privacidade_dados": privacidade,
        "possui_politica_anticorrupcao": anticorrupcao,
        "consta_lista_sancoes": sancoes,
        "noticias_negativas_12m": noticias,
        "quantidade_certificacoes": certificacoes,
        "receita_anual": receita,
    }


def tela_login() -> None:
    st.title("ESG Nexus Enterprise")
    st.markdown("<div class='painel-info'>Plataforma corporativa para avaliação ESG de fornecedores, análise executiva e governança de modelos de Machine Learning.</div>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<div class='titulo-produto'>ESG Nexus</div><div class='subtitulo-produto'>Configuração de acesso</div>", unsafe_allow_html=True)
        st.session_state["url_api"] = st.text_input("URL da API", value=st.session_state.get("url_api", URL_API_PADRAO))
    aba_login, aba_cadastro = st.tabs(["Entrar", "Cadastrar usuário"])
    with aba_login:
        email = st.text_input("E-mail", key="login_email")
        senha = st.text_input("Senha", type="password", key="login_senha")
        if st.button("Entrar", type="primary"):
            try:
                resposta = obter_cliente().login(email, senha)
                st.session_state["token"] = resposta["access_token"]
                st.session_state["usuario"] = resposta["usuario"]
                st.session_state["autenticado"] = True
                st.rerun()
            except (ErroAPI, requests.RequestException) as exc:
                st.error(f"Falha no login: {exc}")
    with aba_cadastro:
        nome = st.text_input("Nome", key="cad_nome")
        email_cad = st.text_input("E-mail", key="cad_email")
        senha_cad = st.text_input("Senha", type="password", key="cad_senha")
        perfil = st.selectbox("Perfil", ["operador_esg", "analista_esg", "gerente_esg", "cientista_dados", "administrador"], key="cad_perfil")
        if st.button("Cadastrar usuário"):
            try:
                obter_cliente().registrar(nome, email_cad, senha_cad, perfil)
                st.success("Usuário cadastrado. Faça login para continuar.")
            except (ErroAPI, requests.RequestException) as exc:
                st.error(f"Falha no cadastro: {exc}")


def render_dashboard_executivo() -> None:
    st.header("Dashboard Executivo ESG")
    st.markdown("<div class='painel-info'>Visão gerencial para priorização de fornecedores, exposição a risco e desempenho por pilar ESG.</div>", unsafe_allow_html=True)
    try:
        payload = dados_executivos()
    except Exception as exc:
        st.error(f"Não foi possível carregar os dados executivos: {exc}")
        return
    kpis = payload.get("kpis", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fornecedores avaliados", kpis.get("total_fornecedores", 0))
    col2.metric("Score ESG médio", kpis.get("score_medio", 0))
    col3.metric("Fornecedores alto risco", kpis.get("alto_risco", 0))
    col4.metric("Probabilidade média de alto risco", f"{kpis.get('probabilidade_ml_media', 0)}%")
    col_a, col_b = st.columns([1.1, 1])
    distribuicao = pd.DataFrame(payload.get("distribuicao_risco", []))
    if not distribuicao.empty and {"nivel_risco", "quantidade"}.issubset(distribuicao.columns):
        with col_a:
            st.plotly_chart(px.pie(distribuicao, names="nivel_risco", values="quantidade", hole=0.45, title="Distribuição por nível de risco"), use_container_width=True)
    elif not distribuicao.empty:
        with col_a:
            st.warning("Os dados de distribuição de risco não possuem o formato esperado.")
    medias = pd.DataFrame(payload.get("medias_pilares", []))
    if not medias.empty:
        with col_b:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=medias["valor"], theta=medias["pilar"], fill="toself", name="Média"))
            fig.update_layout(title="Média por pilar ESG", polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig, use_container_width=True)
    col_c, col_d = st.columns(2)
    top_risco = pd.DataFrame(payload.get("top_risco", []))
    melhores = pd.DataFrame(payload.get("melhores", []))
    with col_c:
        st.subheader("Fornecedores prioritários por risco ML")
        if not top_risco.empty:
            top_risco["probabilidade_ml_alto_risco"] = top_risco["probabilidade_ml_alto_risco"].astype(float) * 100
            st.plotly_chart(px.bar(top_risco, x="probabilidade_ml_alto_risco", y="razao_social", orientation="h", title="Maior probabilidade de alto risco"), use_container_width=True)
            st.dataframe(top_risco, use_container_width=True, hide_index=True)
    with col_d:
        st.subheader("Melhores fornecedores ESG")
        if not melhores.empty:
            st.plotly_chart(px.bar(melhores, x="pontuacao_esg", y="razao_social", orientation="h", title="Ranking por score ESG"), use_container_width=True)
            st.dataframe(melhores, use_container_width=True, hide_index=True)

    # ── Visão consolidada via view ────────────────────────────────────────────
    st.divider()
    st.subheader("Fornecedores — Classificações e Planos de Ação")

    classificacoes = pd.DataFrame(payload.get("fornecedores_classificacoes", []))
    if classificacoes.empty:
        st.info("Nenhum dado disponível. Classifique fornecedores para visualizar esta seção.")
    else:
        tab_cls, tab_planos = st.tabs(["Classificações", "Planos de Ação"])

        with tab_cls:
            _COLS_RESUMO = [
                "razao_social", "setor", "cnpj", "pontuacao_esg", "grade", "level",
                "risco", "impacto", "quadrante", "maturidade_rf", "confianca_rf_high",
                "maturidade_knn", "avaliacao_criado_em",
            ]
            cols_presentes = [c for c in _COLS_RESUMO if c in classificacoes.columns]
            resumo = (
                classificacoes
                .drop_duplicates(subset=["fornecedor_id"])
                [cols_presentes]
                .sort_values("risco", ascending=False)
                .reset_index(drop=True)
            )

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                if {"pontuacao_esg", "risco", "maturidade_rf", "razao_social"}.issubset(resumo.columns):
                    st.plotly_chart(
                        px.scatter(
                            resumo,
                            x="pontuacao_esg", y="risco",
                            color="maturidade_rf", text="razao_social",
                            labels={"pontuacao_esg": "Score ESG (0-100)", "risco": "Risco"},
                            title="Score ESG × Risco por fornecedor",
                        ),
                        use_container_width=True,
                    )
            with col_g2:
                if {"risco", "razao_social", "maturidade_rf"}.issubset(resumo.columns):
                    st.plotly_chart(
                        px.bar(
                            resumo.head(15),
                            x="risco", y="razao_social", orientation="h",
                            color="maturidade_rf",
                            labels={"risco": "Índice de risco", "razao_social": "Fornecedor"},
                            title="Top 15 fornecedores por risco",
                        ),
                        use_container_width=True,
                    )

            st.dataframe(resumo, use_container_width=True, hide_index=True)

        with tab_planos:
            _COLS_PLANO = ["razao_social", "setor", "pilar", "pilar_score", "importancia", "acao"]
            cols_plano = [c for c in _COLS_PLANO if c in classificacoes.columns]
            planos = (
                classificacoes
                .dropna(subset=["plano_id"] if "plano_id" in classificacoes.columns else [])
                [cols_plano]
                .sort_values(
                    ["razao_social", "importancia"] if "importancia" in cols_plano else ["razao_social"],
                    ascending=[True, False],
                )
                .reset_index(drop=True)
            )

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if "pilar" in planos.columns and not planos.empty:
                    dist_pilar = (
                        planos.groupby("pilar", as_index=False)
                        .size()
                        .rename(columns={"size": "total"})
                    )
                    _CORES_PILAR = {"E": "#2ECC71", "S": "#3498DB", "G": "#9B59B6"}
                    st.plotly_chart(
                        px.pie(
                            dist_pilar, names="pilar", values="total",
                            color="pilar", color_discrete_map=_CORES_PILAR,
                            hole=0.4, title="Itens de plano por pilar ESG",
                        ),
                        use_container_width=True,
                    )
            with col_p2:
                if {"pilar", "pilar_score"}.issubset(planos.columns) and not planos.empty:
                    _CORES_PILAR = {"E": "#2ECC71", "S": "#3498DB", "G": "#9B59B6"}
                    st.plotly_chart(
                        px.box(
                            planos, x="pilar", y="pilar_score",
                            color="pilar", color_discrete_map=_CORES_PILAR,
                            labels={"pilar_score": "Score do pilar (0-1000)", "pilar": "Pilar"},
                            title="Distribuição de score por pilar",
                        ),
                        use_container_width=True,
                    )

            st.dataframe(planos, use_container_width=True, hide_index=True)


def render_dashboard_ml() -> None:
    st.header("Dashboard de Machine Learning")
    st.markdown("<div class='painel-info'>Visão técnica para acompanhamento de métricas, comportamento do modelo, explicabilidade e qualidade das previsões.</div>", unsafe_allow_html=True)
    try:
        payload = dados_ml()
    except Exception as exc:
        st.error(f"Não foi possível carregar os dados de ML. Verifique perfil de acesso ou API: {exc}")
        return

    metricas    = payload.get("metricas", {})
    experimento = payload.get("experimento")

    # ── Contexto do treinamento ───────────────────────────────────────────────
    if experimento:
        data_treino = (experimento.get("criado_em") or "")[:10]
        st.caption(f"Treinamento mais recente: **{experimento.get('nome', '-')}** · {data_treino}")
    else:
        st.warning("Nenhum treinamento registrado no banco. Execute `treinar_modelo.py` ou `POST /treinar`.")

    # ── Métricas do Random Forest (treinamento mais recente) ──────────────────
    def _fmt(v) -> str:
        return f"{float(v):.1f}%" if v is not None and v != 0 else "—"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  _fmt(metricas.get("accuracy")))
    col2.metric("Precision", _fmt(metricas.get("precision")))
    col3.metric("Recall",    _fmt(metricas.get("recall")))
    col4.metric("F1 Score",  _fmt(metricas.get("f1")))

    # ── Comparativo RF × KNN ──────────────────────────────────────────────────
    if experimento:
        st.divider()
        st.subheader("Comparativo Random Forest × KNN")

        rf  = experimento.get("rf",  {})
        knn = experimento.get("knn", {})

        _METRICAS_COMP = ["acuracia", "precision", "recall", "f1"]
        _LABELS_COMP   = {"acuracia": "Acurácia", "precision": "Precisão",
                          "recall": "Recall", "f1": "F1 Score"}

        df_comp = pd.DataFrame([
            {
                "Modelo":   modelo,
                **{_LABELS_COMP[m]: (f"{dados.get(m):.1f}%" if dados.get(m) is not None else "—")
                   for m in _METRICAS_COMP},
            }
            for modelo, dados in [("Random Forest", rf), ("KNN", knn)]
        ])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        col_r, col_k = st.columns(2)
        categorias = ["Acurácia", "Precisão", "Recall", "F1 Score"]
        with col_r:
            fig_radar = go.Figure()
            for nome, dados in [("Random Forest", rf), ("KNN", knn)]:
                valores = [dados.get(k) or 0 for k in _METRICAS_COMP]
                fig_radar.add_trace(go.Scatterpolar(
                    r=valores, theta=categorias, fill="toself", name=nome,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Radar de métricas — RF × KNN",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_k:
            df_bar = pd.DataFrame([
                {"Métrica": _LABELS_COMP[m], "Random Forest": rf.get(m) or 0,
                 "KNN": knn.get(m) or 0}
                for m in _METRICAS_COMP
            ])
            fig_bar = px.bar(
                df_bar.melt(id_vars="Métrica", var_name="Modelo", value_name="Valor (%)"),
                x="Métrica", y="Valor (%)", color="Modelo", barmode="group",
                range_y=[0, 105], title="Métricas por modelo (%)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Gráficos de comportamento do modelo ───────────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)
    scores = pd.DataFrame(payload.get("distribuicao_scores", []))
    if not scores.empty:
        with col_a:
            st.plotly_chart(px.histogram(scores, x="pontuacao_esg", color="nivel_risco",
                                         nbins=20, title="Distribuição dos scores ESG"),
                            use_container_width=True)
    dispersao = pd.DataFrame(payload.get("dispersao", []))
    if not dispersao.empty:
        with col_b:
            st.plotly_chart(px.scatter(dispersao, x="pontuacao_esg",
                                       y="probabilidade_ml_alto_risco",
                                       color="nivel_risco", size="pontuacao_governanca",
                                       title="Score ESG × Probabilidade de alto risco"),
                            use_container_width=True)

    col_c, col_d = st.columns(2)
    importancia = pd.DataFrame(payload.get("feature_importance", []))
    if not importancia.empty:
        with col_c:
            st.plotly_chart(px.bar(importancia.sort_values("importancia"),
                                   x="importancia", y="variavel", orientation="h",
                                   title="Importância das variáveis"),
                            use_container_width=True)
    if not dispersao.empty:
        with col_d:
            correlacao = dispersao[["pontuacao_esg", "probabilidade_ml_alto_risco",
                                    "pontuacao_ambiental", "pontuacao_social",
                                    "pontuacao_governanca"]].corr()
            st.plotly_chart(px.imshow(correlacao, text_auto=True,
                                      title="Correlação entre variáveis ESG e ML"),
                            use_container_width=True)

    st.subheader("Amostra analisada pelo modelo")
    if not scores.empty:
        st.dataframe(scores, use_container_width=True, hide_index=True)


def render_fornecedores() -> None:
    st.header("Cadastro e consulta de fornecedores")
    with st.expander("Novo fornecedor", expanded=False):
        fornecedor = formulario_fornecedor("cad_")
        if st.button("Salvar fornecedor", type="primary"):
            try:
                resposta = obter_cliente().cadastrar_fornecedor(fornecedor)
                st.success(f"Fornecedor cadastrado: {resposta.get('codigo_fornecedor')}")
            except Exception as exc:
                st.error(f"Falha ao cadastrar fornecedor: {exc}")
    st.subheader("Fornecedores cadastrados")
    try:
        st.dataframe(pd.DataFrame(obter_cliente().listar_fornecedores()), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Falha ao listar fornecedores: {exc}")


def render_classificacao() -> None:
    st.header("Classificação ESG e Explicabilidade de IA")
    fornecedor = formulario_fornecedor("cls_")
    col1, col2 = st.columns([1, 1])
    if col1.button("Classificar fornecedor", type="primary"):
        try:
            st.session_state["ultima_classificacao"] = obter_cliente().classificar(fornecedor)
        except Exception as exc:
            st.error(f"Falha na classificação: {exc}")
    if col2.button("Gerar explicabilidade"):
        try:
            st.session_state["ultima_explicacao"] = obter_cliente().explicar(fornecedor)
        except Exception as exc:
            st.error(f"Falha na explicabilidade: {exc}")
    avaliacao = st.session_state.get("ultima_classificacao")
    if avaliacao:
        st.subheader("Resultado da classificação")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Score ESG", avaliacao.get("pontuacao_esg"))
        col_b.metric("Nível de risco", avaliacao.get("nivel_risco"))
        col_c.metric("Probabilidade de alto risco", f"{float(avaliacao.get('probabilidade_ml_alto_risco', 0))*100:.1f}%")
        col_d.metric("Recomendação", avaliacao.get("recomendacao", "-"))
        pilares = pd.DataFrame([{"pilar": "Ambiental", "valor": avaliacao.get("pontuacao_ambiental", 0)}, {"pilar": "Social", "valor": avaliacao.get("pontuacao_social", 0)}, {"pilar": "Governança", "valor": avaliacao.get("pontuacao_governanca", 0)}])
        st.plotly_chart(px.bar(pilares, x="pilar", y="valor", range_y=[0, 100], title="Pontuação por pilar"), use_container_width=True)
        st.write("Motivos:")
        for motivo in avaliacao.get("motivos", []):
            st.write(f"- {motivo}")
    explicacao = st.session_state.get("ultima_explicacao")
    if explicacao:
        st.subheader("Explicabilidade da decisão")
        fatores = pd.DataFrame(explicacao.get("fatores", []))
        if not fatores.empty:
            coluna_fator = "fator" if "fator" in fatores.columns else fatores.columns[0]
            coluna_impacto = "impacto" if "impacto" in fatores.columns else fatores.columns[1]
            st.plotly_chart(px.bar(fatores, x=coluna_impacto, y=coluna_fator, orientation="h", title="Fatores que influenciaram a decisão"), use_container_width=True)
            st.dataframe(fatores, use_container_width=True, hide_index=True)


def render_importacao_lote() -> None:
    st.header("Importação e Classificação em Lote")
    st.markdown("<div class='painel-info'>Importe uma planilha XLSX ou CSV de fornecedores, valide os dados, execute a classificação ESG em lote e baixe o resultado para análise e auditoria.</div>", unsafe_allow_html=True)

    st.download_button(
        "Baixar modelo de planilha",
        data=gerar_modelo_planilha(),
        file_name="modelo_importacao_fornecedores_esg.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    arquivo = st.file_uploader("Planilha de fornecedores", type=["xlsx", "xls", "csv"])
    if arquivo is None:
        st.info("Selecione uma planilha para iniciar a validação.")
        return

    try:
        df_original = carregar_planilha_fornecedores(arquivo)
    except Exception as exc:
        st.error(f"Não foi possível ler a planilha: {exc}")
        return

    st.subheader("Pré-visualização")
    st.dataframe(df_original.head(20), use_container_width=True, hide_index=True)

    df_validado, erros = validar_e_normalizar_importacao(df_original)
    if erros:
        st.error("A planilha possui inconsistências. Corrija os pontos abaixo antes de classificar.")
        st.dataframe(pd.DataFrame({"inconsistencia": erros}), use_container_width=True, hide_index=True)
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Linhas válidas", len(df_validado))
    col2.metric("Fornecedores únicos", df_validado["codigo_fornecedor"].nunique())
    col3.metric("Setores", df_validado["setor"].nunique())

    st.subheader("Dados normalizados para classificação")
    st.dataframe(df_validado.head(50), use_container_width=True, hide_index=True)

    if st.button("Classificar fornecedores em lote", type="primary"):
        fornecedores = dataframe_para_registros_json(df_validado)
        try:
            resposta = obter_cliente().classificar_lote(fornecedores)
            st.session_state["resultado_lote"] = resposta
            st.success(f"Classificação concluída. Processados: {resposta.get('total_processados', 0)}. Erros: {resposta.get('total_erros', 0)}.")
        except Exception as exc:
            st.error(f"Falha ao classificar fornecedores em lote: {exc}")

    resposta = st.session_state.get("resultado_lote")
    if not resposta:
        return

    df_resultado = pd.DataFrame(resposta.get("resultados", []))
    df_erros = pd.DataFrame(resposta.get("erros", []))
    if df_resultado.empty:
        st.warning("Nenhum resultado retornado pela classificação.")
        return

    st.subheader("Resultado da classificação")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total processado", resposta.get("total_processados", 0))
    col_b.metric("Score médio", round(float(df_resultado["pontuacao_esg"].mean()), 2))
    col_c.metric("Alto risco", int((df_resultado["nivel_risco"].astype(str).str.lower() == "alto").sum()))
    col_d.metric("Erros", resposta.get("total_erros", 0))

    graf1, graf2 = st.columns(2)
    with graf1:
        dist = df_resultado.groupby("nivel_risco", as_index=False).size().rename(columns={"size": "quantidade"})
        st.plotly_chart(px.pie(dist, names="nivel_risco", values="quantidade", hole=0.45, title="Distribuição por nível de risco"), use_container_width=True)
    with graf2:
        st.plotly_chart(px.bar(df_resultado.sort_values("pontuacao_esg", ascending=True).tail(15), x="pontuacao_esg", y="razao_social", orientation="h", title="Ranking ESG dos fornecedores"), use_container_width=True)

    st.dataframe(df_resultado, use_container_width=True, hide_index=True)
    if not df_erros.empty:
        st.subheader("Erros de processamento")
        st.dataframe(df_erros, use_container_width=True, hide_index=True)

    st.download_button(
        "Baixar resultado em Excel",
        data=gerar_excel_resultado(df_resultado, df_erros),
        file_name="classificacao_esg_fornecedores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Explicabilidade individual")
    opcoes = df_resultado["codigo_fornecedor"].astype(str) + " - " + df_resultado["razao_social"].astype(str)
    selecionado = st.selectbox("Fornecedor para detalhamento", opcoes.tolist())
    codigo = selecionado.split(" - ", 1)[0]
    linha = df_resultado[df_resultado["codigo_fornecedor"].astype(str) == codigo].iloc[0].to_dict()
    colunas_metricas = st.columns(4)
    colunas_metricas[0].metric("Score ESG", linha.get("pontuacao_esg"))
    colunas_metricas[1].metric("Nível de risco", linha.get("nivel_risco"))
    colunas_metricas[2].metric("Probabilidade alto risco", f"{float(linha.get('probabilidade_ml_alto_risco', 0))*100:.1f}%")
    colunas_metricas[3].metric("Recomendação", linha.get("recomendacao", "-"))
    motivos = linha.get("motivos", [])
    if isinstance(motivos, list):
        st.write("Motivos da classificação:")
        for motivo in motivos:
            st.write(f"- {motivo}")


def render_dashboard_estatistico() -> None:
    st.title("Dashboard Estatístico")
    st.caption("Análise estatística dos dados do banco de dados ESG Nexus — médias, distribuições, correlações e tendências.")

    # ── Filtros ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
    with col1:
        dt_inicio = st.date_input("Data inicial", value=None, key="est_di")
    with col2:
        dt_fim = st.date_input("Data final", value=None, key="est_df")
    with col3:
        setores_disp = st.session_state.get("est_setores", [])
        setor = st.selectbox(
            "Setor", [None] + setores_disp,
            format_func=lambda x: "Todos os setores" if x is None else x,
            key="est_setor",
        )
    with col4:
        maturidade = st.selectbox(
            "Maturidade RF", [None, "High", "Medium", "Low"],
            format_func=lambda x: "Todas" if x is None else x,
            key="est_mat",
        )

    # ── Buscar dados ───────────────────────────────────────────────────────────
    try:
        dados = obter_cliente().dashboard_estatistico(
            data_inicio=str(dt_inicio) if dt_inicio else None,
            data_fim=str(dt_fim) if dt_fim else None,
            setor=setor,
            maturidade=maturidade,
        )
    except ErroAPI as exc:
        st.error(f"Erro ao carregar dados estatísticos: {exc}")
        return

    st.session_state["est_setores"] = dados.get("setores_disponiveis", [])

    kpis = dados.get("kpis", {})
    if not kpis:
        st.info("Nenhuma avaliação encontrada para os filtros selecionados.")
        return

    # ── KPIs ───────────────────────────────────────────────────────────────────
    st.markdown("### Indicadores Resumidos")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de Avaliações", kpis.get("total_avaliacoes", 0))
    c2.metric("Fornecedores", kpis.get("total_fornecedores", 0))
    c3.metric("Score ESG Médio", f"{kpis.get('score_medio', 0):.2f}")
    c4.metric("% Alto Risco", f"{kpis.get('percentual_alto_risco', 0):.1f}%")
    c5.metric("% Maturidade High", f"{kpis.get('percentual_maturidade_high', 0):.1f}%")

    # ── Tabela de estatísticas descritivas ─────────────────────────────────────
    st.markdown("### Estatísticas Descritivas por Pilar")
    st.caption("Média, mediana, moda, desvio padrão, variância, mínimo, máximo e quartis.")
    estatisticas = dados.get("estatisticas", {})
    _labels_est = {
        "ambiental":  "Ambiental (0–100)",
        "social":     "Social (0–100)",
        "governanca": "Governança (0–100)",
        "total_esg":  "Score ESG Ponderado",
        "risco":      "Risco",
    }
    linhas_est = []
    for k, label in _labels_est.items():
        s = estatisticas.get(k, {})
        if s:
            linhas_est.append({
                "Variável":       label,
                "n":              s.get("total"),
                "Média":          s.get("media"),
                "Mediana":        s.get("mediana"),
                "Moda":           s.get("moda"),
                "Desvio Padrão":  s.get("desvio_padrao"),
                "Variância":      s.get("variancia"),
                "Mín":            s.get("minimo"),
                "Máx":            s.get("maximo"),
                "Q1":             s.get("q1"),
                "Q3":             s.get("q3"),
            })
    if linhas_est:
        st.dataframe(pd.DataFrame(linhas_est), use_container_width=True, hide_index=True)

    # ── Gráficos ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Gráficos Estatísticos")

    registros = dados.get("registros_detalhe", [])
    df_det = pd.DataFrame(registros) if registros else pd.DataFrame()

    # Gráfico 1 + 2
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gráfico 1 — Box Plot: Distribuição por Pilar**")
        st.caption("Média, mediana, Q1/Q3 e outliers de cada pilar ESG.")
        if not df_det.empty:
            df_melt = pd.melt(
                df_det, id_vars=["setor", "nivel_risco"],
                value_vars=["ambiental", "social", "governanca"],
                var_name="pilar", value_name="score",
            )
            df_melt["pilar"] = df_melt["pilar"].map(
                {"ambiental": "Ambiental", "social": "Social", "governanca": "Governança"})
            fig = px.box(
                df_melt, x="pilar", y="score", color="pilar",
                color_discrete_map={
                    "Ambiental": "#2ecc71", "Social": "#3498db", "Governança": "#9b59b6"},
                labels={"score": "Score (0–100)", "pilar": "Pilar"},
            )
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Gráfico 2 — Histograma: Frequência do Score ESG**")
        st.caption("Distribuição de frequência relativa e absoluta do Score ESG ponderado.")
        if not df_det.empty:
            fig = px.histogram(
                df_det, x="pontuacao_esg", nbins=20,
                color_discrete_sequence=["#2ecc71"],
                labels={"pontuacao_esg": "Score ESG", "count": "Frequência"},
            )
            fig.update_layout(bargap=0.05, yaxis_title="Frequência", margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Gráfico 3 + 4
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gráfico 3 — Tendência Temporal: Score ESG e Risco por Mês**")
        st.caption("Evolução do Score ESG médio (verde) e Risco médio (vermelho) ao longo do tempo.")
        serie = dados.get("serie_temporal", [])
        if serie:
            df_s = pd.DataFrame(serie)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_s["periodo"], y=df_s["score_medio"],
                name="Score ESG Médio", mode="lines+markers",
                line=dict(color="#2ecc71", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=df_s["periodo"], y=df_s["risco_medio"],
                name="Risco Médio", mode="lines+markers",
                line=dict(color="#e74c3c", width=2), yaxis="y2",
            ))
            fig.update_layout(
                xaxis_title="Período",
                yaxis=dict(title="Score ESG", color="#2ecc71", side="left"),
                yaxis2=dict(title="Risco", color="#e74c3c", side="right", overlaying="y"),
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dados temporais insuficientes para o período selecionado.")

    with col2:
        st.markdown("**Gráfico 4 — Radar: Média vs Mediana por Pilar**")
        st.caption("Comparativo entre as medidas de tendência central por pilar ESG.")
        if estatisticas:
            _pilares = ["Ambiental", "Social", "Governança"]
            _keys    = ["ambiental", "social", "governanca"]
            medias   = [estatisticas.get(k, {}).get("media",   0) or 0 for k in _keys]
            medianas = [estatisticas.get(k, {}).get("mediana", 0) or 0 for k in _keys]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=medias + [medias[0]], theta=_pilares + [_pilares[0]],
                fill="toself", name="Média", line_color="#2ecc71",
            ))
            fig.add_trace(go.Scatterpolar(
                r=medianas + [medianas[0]], theta=_pilares + [_pilares[0]],
                fill="toself", name="Mediana", line_color="#e67e22", opacity=0.7,
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], tickfont_size=10)),
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Gráfico 5 + 6
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gráfico 5 — Score Médio por Setor (com Desvio Padrão)**")
        st.caption("Média e desvio padrão do Score ESG por setor de atividade econômica.")
        por_setor = dados.get("por_setor", [])
        if por_setor:
            df_ps = pd.DataFrame(por_setor)
            df_ps["desvio_padrao"] = df_ps["desvio_padrao"].fillna(0)
            fig = go.Figure(go.Bar(
                x=df_ps["setor"], y=df_ps["score_medio"],
                error_y=dict(type="data", array=df_ps["desvio_padrao"].tolist(), visible=True),
                marker_color="#2ecc71",
                text=df_ps["score_medio"].round(1), textposition="outside",
            ))
            fig.update_layout(
                xaxis_tickangle=-35,
                xaxis_title="Setor", yaxis_title="Score ESG",
                margin=dict(t=10, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Gráfico 6 — Distribuição por Nível de Risco**")
        st.caption("Proporção de avaliações em cada nível de risco (baixo / médio / alto).")
        dist_risco = dados.get("distribuicao_risco", [])
        if dist_risco:
            df_dr = pd.DataFrame(dist_risco)
            _CORES_R = {"alto": "#e74c3c", "médio": "#f39c12", "baixo": "#2ecc71"}
            fig = px.pie(
                df_dr, names="nivel_risco", values="quantidade",
                color="nivel_risco", color_discrete_map=_CORES_R, hole=0.4,
            )
            fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{value} (%{percent})",
                hovertemplate="%{label}: %{value} avaliações (%{percent})<extra></extra>",
            )
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Gráfico 7 + 8
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gráfico 7 — Heatmap: Correlação entre Variáveis ESG**")
        st.caption("Coeficiente de correlação de Pearson entre os pilares, score ponderado, risco e impacto.")
        correlacoes = dados.get("correlacoes", {})
        if correlacoes and correlacoes.get("labels") and correlacoes.get("matrix"):
            fig = px.imshow(
                correlacoes["matrix"],
                x=correlacoes["labels"], y=correlacoes["labels"],
                text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            )
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Gráfico 8 — Planos de Ação por Pilar ESG**")
        st.caption("Total de ações de melhoria recomendadas para cada pilar (E / S / G).")
        planos = dados.get("planos_por_pilar", [])
        if planos:
            df_pl = pd.DataFrame(planos)
            _CORES_P  = {"E": "#2ecc71", "S": "#3498db", "G": "#9b59b6"}
            _LABELS_P = {"E": "Ambiental", "S": "Social", "G": "Governança"}
            df_pl["pilar_label"] = df_pl["pilar"].map(_LABELS_P).fillna(df_pl["pilar"])
            fig = go.Figure(go.Bar(
                x=df_pl["pilar_label"], y=df_pl["total"],
                marker_color=[_CORES_P.get(p, "#95a5a6") for p in df_pl["pilar"]],
                text=df_pl["total"], textposition="outside", name="Ações",
            ))
            fig.update_layout(
                xaxis_title="Pilar", yaxis_title="Nº de Ações",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum plano de ação disponível para o filtro selecionado.")

    # ── Histórico ML (quando há mais de 1 treinamento) ─────────────────────────
    historico_ml = dados.get("historico_ml", [])
    if len(historico_ml) > 1:
        st.markdown("---")
        st.markdown("**Evolução Histórica dos Modelos ML**")
        st.caption("Acurácia (%) e F1-Score (%) de Random Forest e KNN ao longo dos treinamentos.")
        df_ml = pd.DataFrame(historico_ml)
        df_ml["label"] = df_ml["nome"].str.replace("treino_", "", regex=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_ml["label"], y=df_ml["rf_acuracia"],
            name="RF – Acurácia %", mode="lines+markers",
            line=dict(color="#2ecc71", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_ml["label"], y=df_ml["knn_acuracia"],
            name="KNN – Acurácia %", mode="lines+markers",
            line=dict(color="#3498db", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_ml["label"], y=df_ml["rf_f1"],
            name="RF – F1 %", mode="lines+markers",
            line=dict(color="#27ae60", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df_ml["label"], y=df_ml["knn_f1"],
            name="KNN – F1 %", mode="lines+markers",
            line=dict(color="#2980b9", width=2, dash="dot"),
        ))
        fig.update_layout(
            yaxis_title="Valor (%)", xaxis_tickangle=-30,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=10, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Dados detalhados ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Dados Detalhados das Avaliações", expanded=False):
        if not df_det.empty:
            st.dataframe(
                df_det.rename(columns={
                    "setor": "Setor", "ambiental": "Ambiental", "social": "Social",
                    "governanca": "Governança", "pontuacao_esg": "Score ESG",
                    "risco": "Risco", "impacto": "Impacto",
                    "nivel_risco": "Nível Risco", "maturidade_rf": "Maturidade RF",
                    "grade": "Grade", "periodo": "Período",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Sem dados para exibir.")


def render_operacao_ml() -> None:
    st.header("Operação de ML, Airflow e MLflow")
    st.markdown("<div class='painel-info'>Tela operacional para acompanhar a orquestração do pipeline, rastreamento de experimentos e artefatos do modelo.</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Airflow DAG", "dag_treinamento_esg")
    col2.metric("MLflow Tracking", os.getenv("MLFLOW_TRACKING_URL", os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")))
    col3.metric("Modelo ativo", "modelo_fornecedores_esg.joblib")
    st.subheader("Comandos úteis")
    st.code("""# API
uvicorn esg_ml.interfaces.api.principal:app --reload

# Streamlit
streamlit run frontend/app.py

# Airflow
airflow dags trigger dag_treinamento_esg

# MLflow
mlflow ui --host 0.0.0.0 --port 5000
""", language="bash")


def app() -> None:
    if not st.session_state.get("autenticado"):
        tela_login()
        return
    usuario = st.session_state.get("usuario", {})
    with st.sidebar:
        st.markdown("<div class='titulo-produto'>ESG Nexus</div><div class='subtitulo-produto'>Plataforma Enterprise</div>", unsafe_allow_html=True)
        st.write(f"**Usuário:** {usuario.get('nome', '-')}")
        st.write(f"**Perfil:** {usuario.get('perfil', '-')}")
        st.session_state["url_api"] = st.text_input("URL da API", value=st.session_state.get("url_api", URL_API_PADRAO))
        opcoes_menu = paginas_permitidas(str(usuario.get("perfil", "")))
        pagina = st.radio("Navegação", opcoes_menu)
        if st.button("Sair"):
            for chave in ["autenticado", "token", "usuario", "ultima_classificacao", "ultima_explicacao"]:
                st.session_state.pop(chave, None)
            st.rerun()
    if pagina == "Dashboard Executivo ESG":
        render_dashboard_executivo()
    elif pagina == "Dashboard Estatístico":
        render_dashboard_estatistico()
    elif pagina == "Dashboard de Machine Learning":
        render_dashboard_ml()
    elif pagina == "Fornecedores":
        render_fornecedores()
    elif pagina == "Importação e Classificação em Lote":
        render_importacao_lote()
    elif pagina == "Classificação e Explicabilidade":
        render_classificacao()
    else:
        render_operacao_ml()


if __name__ == "__main__":
    app()
