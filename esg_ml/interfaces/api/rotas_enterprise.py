# esg_ml/interfaces/api/rotas_enterprise.py
# v3: FornecedorEntrada com 17 métricas PT → scores ESG calculados internamente.
# Saída inclui campos PT (pontuacao_esg, nivel_risco, motivos…) para o frontend.

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from esg_ml.adaptadores.saida.repositorio_modelo_joblib import RepositorioModeloJoblib
from esg_ml.aplicacao.servico_avaliacao import ServicoAvaliacao
from esg_ml.dominio.entidades.empresa import Empresa, ScoreESG
from esg_ml.dominio.entidades.diagnostico import DiagnosticoESG
from esg_ml.infraestrutura.banco_dados import obter_sessao
from esg_ml.infraestrutura.configuracoes import Configuracoes
from esg_ml.infraestrutura.modelos_banco import (
    AvaliacaoBanco, ExperimentoMLBanco, FornecedorBanco, PlanoAcaoBanco, UsuarioBanco)
from esg_ml.interfaces.api.dependencias_auth import exigir_perfil, obter_usuario_atual
from esg_ml.interfaces.api.esquemas import (
    AvaliacaoSaida,
    ClassificacaoLoteEntrada,
    FornecedorEntrada,
    FornecedorSaida,
    HistoricoFornecedorSaida,
    HistoricoItemSaida,
    PlanoAcaoSaida,
    ResultadoClassificacaoLote,
)

roteador   = APIRouter(tags=['Enterprise'])
conf       = Configuracoes()
repositorio = RepositorioModeloJoblib(conf.diretorio_artefatos)


# ── Mapeamento setor PT → industry EN (LabelEncoder Kaggle) ──────────────────

_MAPA_SETOR: dict[str, str] = {
    'tecnologia':          'Technology',
    'financeiro':          'Financial Services',
    'saude':               'Healthcare',
    'saúde':               'Healthcare',
    'energia':             'Energy',
    'industria':           'Industrials',
    'indústria':           'Industrials',
    'consumo':             'Consumer Discretionary',
    'varejo':              'Consumer Staples',
    'materiais':           'Materials',
    'servicos':            'Services',
    'serviços':            'Services',
    'logistica':           'Industrials',
    'logística':           'Industrials',
    'agronegocio':         'Consumer Staples',
    'agronegócio':         'Consumer Staples',
    'mineracao':           'Materials',
    'mineração':           'Materials',
    'telecomunicacoes':    'Communication Services',
    'telecomunicações':    'Communication Services',
    'imobiliario':         'Real Estate',
    'imobiliário':         'Real Estate',
    'utilidades':          'Utilities',
    'limpeza':             'Services',
    'alimentos':           'Consumer Staples',
    'transporte':          'Industrials',
}

_NOME_PILAR: dict[str, str] = {'E': 'Ambiental', 'S': 'Social', 'G': 'Governança'}


# ── Conversão de métricas PT → scores ESG ────────────────────────────────────

def _calcular_scores_esg(entrada: FornecedorEntrada) -> tuple[int, int, int, str]:
    """17 métricas de domínio PT → (env_score, social_score, gov_score, industry)."""

    # Ambiental (0-1000)
    e  = 250.0 if entrada.possui_politica_ambiental else 0.0
    e += min(entrada.percentual_energia_renovavel   * 2.5,  250.0)
    e += min(entrada.percentual_reciclagem_residuos * 2.0,  200.0)
    e += max(0.0, 200.0 * (1.0 - min(entrada.emissoes_carbono_ton / 5000.0, 1.0)))
    e += min(entrada.quantidade_certificacoes       * 20.0, 100.0)

    # Social (0-1000)
    s  = 250.0 if entrada.possui_programa_diversidade else 0.0
    s += max(0.0, 400.0 - entrada.incidentes_trabalhistas_12m * 40.0)
    s += max(0.0, 350.0 - entrada.noticias_negativas_12m      * 35.0)

    # Governança (0-1000)
    g  = 300.0 if entrada.possui_politica_privacidade_dados else 0.0
    g += 300.0 if entrada.possui_politica_anticorrupcao      else 0.0
    g += max(0.0, 200.0 - entrada.noticias_negativas_12m * 20.0)
    g += min(entrada.quantidade_certificacoes * 20.0, 200.0)
    if entrada.consta_lista_sancoes:
        g *= 0.1

    industry = _MAPA_SETOR.get(entrada.setor.lower().strip(), 'Services')
    return (int(min(max(e, 0.0), 1000.0)),
            int(min(max(s, 0.0), 1000.0)),
            int(min(max(g, 0.0), 1000.0)),
            industry)


def _nivel_risco(risco: float) -> str:
    if risco > 60:
        return 'alto'
    if risco > 30:
        return 'medio'
    return 'baixo'


# ── Helpers de conversão ──────────────────────────────────────────────────────

def _dominio(entrada: FornecedorEntrada) -> Empresa:
    """FornecedorEntrada (17 métricas PT) → Empresa (domínio ML)."""
    env, soc, gov, industry = _calcular_scores_esg(entrada)
    return Empresa(
        name=entrada.razao_social,
        industry=industry,
        scores=ScoreESG(environment_score=env, social_score=soc, governance_score=gov),
        cnpj=entrada.cnpj or None,
    )


def _saida(d: DiagnosticoESG, entrada: FornecedorEntrada | None = None) -> AvaliacaoSaida:
    """DiagnosticoESG → AvaliacaoSaida com campos PT para o frontend."""
    dados = d.to_dict()

    # Identificação PT
    if entrada is not None:
        dados['codigo_fornecedor'] = entrada.codigo_fornecedor
        dados['razao_social']      = entrada.razao_social
        dados['setor']             = entrada.setor

    # Pontuação 0-100
    dados['pontuacao_ambiental']  = d.environment_score // 10
    dados['pontuacao_social']     = d.social_score      // 10
    dados['pontuacao_governanca'] = d.governance_score  // 10
    dados['pontuacao_esg']        = round(d.score_ponderado / 10.0, 1)

    # Nível de risco e recomendação
    nr = _nivel_risco(d.risco)
    dados['nivel_risco']  = nr
    dados['recomendacao'] = {
        'alto':  'Requer plano de ação ESG imediato',
        'medio': 'Aprovar com plano de melhoria ESG',
        'baixo': 'Aprovar',
    }[nr]

    dados['probabilidade_ml_alto_risco'] = round(d.confianca_rf_high / 100.0, 3)

    # Motivos derivados do plano de ação
    motivos = [
        f"Pilar {_NOME_PILAR.get(item.pilar, item.pilar)} requer atenção (score {item.score}/1000)"
        for item in d.plano_acao[:3]
    ]
    dados['motivos'] = motivos or [
        f"Desempenho ESG {nr} — pontuação {dados['pontuacao_esg']}/100"
    ]

    dados['plano_acao'] = [
        PlanoAcaoSaida(pilar=item.pilar, score=item.score,
                       importancia=round(item.importancia, 4), acao=item.acao)
        for item in d.plano_acao
    ]
    return AvaliacaoSaida(**dados)


# ── Helpers de persistência ───────────────────────────────────────────────────

def _persistir_fornecedor(sessao: Session, entrada: FornecedorEntrada) -> FornecedorBanco:
    """Upsert de FornecedorBanco por CNPJ (fallback: razao_social)."""
    env, soc, gov, industry = _calcular_scores_esg(entrada)

    if entrada.cnpj:
        existente = sessao.query(FornecedorBanco).filter(
            FornecedorBanco.cnpj == entrada.cnpj).first()
    else:
        existente = sessao.query(FornecedorBanco).filter(
            FornecedorBanco.name == entrada.razao_social).first()

    if existente is None:
        forn = FornecedorBanco(
            name=entrada.razao_social,
            industry=industry,
            environment_score=env,
            social_score=soc,
            governance_score=gov,
            cnpj=entrada.cnpj or None,
        )
        sessao.add(forn)
        return forn

    existente.name              = entrada.razao_social
    existente.industry          = industry
    existente.environment_score = env
    existente.social_score      = soc
    existente.governance_score  = gov
    return existente


def _persistir_avaliacao(sessao: Session, d: DiagnosticoESG,
                          fornecedor_id: int | None = None) -> AvaliacaoBanco:
    dados = d.to_dict()
    dados['fornecedor_id'] = fornecedor_id
    av = AvaliacaoBanco(**dados)
    sessao.add(av)
    return av


def _persistir_plano_acao(sessao: Session, d: DiagnosticoESG,
                           avaliacao_id: int,
                           fornecedor_id: int | None = None) -> None:
    for item in d.plano_acao:
        sessao.add(PlanoAcaoBanco(
            avaliacao_id=avaliacao_id,
            fornecedor_id=fornecedor_id,
            pilar=item.pilar,
            score=item.score,
            importancia=item.importancia,
            acao=item.acao,
        ))


def _classificar_e_persistir(
    sessao: Session,
    entrada: FornecedorEntrada,
    servico: ServicoAvaliacao,
    *,
    usuario_id: int | None = None,
    usuario_email: str | None = None,
    lote_id: str | None = None,
    operacao: str = 'importacao',
) -> AvaliacaoSaida:
    d        = servico.avaliar_um(_dominio(entrada))
    forn     = _persistir_fornecedor(sessao, entrada)
    sessao.flush()
    if usuario_id is not None:
        sessao.execute(
            text("CALL sp_registrar_importacao(:fid, :rs, :cnpj, :uid, :uemail, :op, :lote, NULL)"),
            {'fid': forn.id, 'rs': forn.name, 'cnpj': forn.cnpj,
             'uid': usuario_id, 'uemail': usuario_email, 'op': operacao, 'lote': lote_id},
        )
    av_banco = _persistir_avaliacao(sessao, d, forn.id)
    sessao.flush()
    _persistir_plano_acao(sessao, d, av_banco.id, forn.id)
    return _saida(d, entrada)


# ── Treino ────────────────────────────────────────────────────────────────────

@roteador.post('/treinar',
               dependencies=[Depends(exigir_perfil('administrador', 'cientista_dados'))])
def treinar(sessao: Session = Depends(obter_sessao)) -> dict:
    """CRISP-DM Fases 2–6: treina modelos e persiste métricas no banco para o dashboard ML."""
    from datetime import datetime, timezone
    from esg_ml.aplicacao.servico_treinamento import ServicoTreinamento
    from esg_ml.dominio.servicos.avaliacao import ModeloInsuficienteError
    try:
        resultado = ServicoTreinamento(repositorio).treinar()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModeloInsuficienteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    exp = ExperimentoMLBanco(
        nome_execucao=f"treino_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        knn_acuracia=resultado.get('knn_acuracia', 0.0),
        knn_f1_medium=resultado.get('knn_f1_medium', 0.0),
        knn_precision=resultado.get('knn_precision'),
        knn_recall=resultado.get('knn_recall'),
        rf_acuracia=resultado.get('rf_acuracia', 0.0),
        rf_f1_medium=resultado.get('rf_f1_medium', 0.0),
        rf_precision=resultado.get('rf_precision'),
        rf_recall=resultado.get('rf_recall'),
        knn_params=str(resultado.get('knn_params', '')),
        rf_params=str(resultado.get('rf_params', '')),
    )
    sessao.add(exp)
    sessao.commit()

    return resultado


# ── Cadastro de fornecedor (sem classificação) ────────────────────────────────

@roteador.post('/fornecedores', response_model=FornecedorSaida)
def cadastrar_fornecedor(
    entrada: FornecedorEntrada,
    sessao: Session = Depends(obter_sessao),
    usuario: UsuarioBanco = Depends(obter_usuario_atual),
) -> FornecedorSaida:
    """Cadastra ou atualiza fornecedor sem executar classificação ML."""
    forn = _persistir_fornecedor(sessao, entrada)
    sessao.flush()
    sessao.execute(
        text("CALL sp_registrar_importacao(:fid, :rs, :cnpj, :uid, :uemail, 'criacao', NULL, NULL)"),
        {'fid': forn.id, 'rs': forn.name, 'cnpj': forn.cnpj,
         'uid': usuario.id, 'uemail': usuario.email},
    )
    sessao.commit()
    total = sessao.query(AvaliacaoBanco).filter(
        AvaliacaoBanco.fornecedor_id == forn.id).count()
    return FornecedorSaida(
        id=forn.id,
        codigo_fornecedor=entrada.codigo_fornecedor,
        razao_social=entrada.razao_social,
        name=forn.name,
        industry=forn.industry,
        environment_score=forn.environment_score,
        social_score=forn.social_score,
        governance_score=forn.governance_score,
        cnpj=forn.cnpj,
        criado_em=forn.criado_em,
        atualizado_em=forn.atualizado_em,
        total_avaliacoes=total,
    )


# ── Explicabilidade ──────────────────────────────────────────────────────────

def _fatores_explicabilidade(entrada: FornecedorEntrada) -> list[dict]:
    """Contribuição de cada métrica de entrada para o score ESG (escala 0-100)."""
    fatores = []

    # Ambiental — baseado em _calcular_scores_esg (escala 0-1000 → /10)
    if entrada.possui_politica_ambiental:
        fatores.append({'fator': 'Política ambiental', 'impacto': 25, 'direcao': 'positivo'})
    energia_imp = round(min(entrada.percentual_energia_renovavel * 0.25, 25), 1)
    if energia_imp > 0:
        fatores.append({'fator': 'Energia renovável', 'impacto': energia_imp, 'direcao': 'positivo'})
    reciclagem_imp = round(min(entrada.percentual_reciclagem_residuos * 0.2, 20), 1)
    if reciclagem_imp > 0:
        fatores.append({'fator': 'Reciclagem de resíduos', 'impacto': reciclagem_imp, 'direcao': 'positivo'})
    emissoes_pen = round(min(entrada.emissoes_carbono_ton / 5000.0, 1.0) * 20, 1)
    if emissoes_pen > 0:
        fatores.append({'fator': 'Emissões de carbono', 'impacto': -emissoes_pen, 'direcao': 'negativo'})
    cert_e = round(min(entrada.quantidade_certificacoes * 2.0, 10), 1)
    if cert_e > 0:
        fatores.append({'fator': 'Certificações ESG', 'impacto': cert_e, 'direcao': 'positivo'})

    # Social
    if entrada.possui_programa_diversidade:
        fatores.append({'fator': 'Programa de diversidade', 'impacto': 25, 'direcao': 'positivo'})
    incidentes_pen = round(min(entrada.incidentes_trabalhistas_12m * 4.0, 40), 1)
    if incidentes_pen > 0:
        fatores.append({'fator': 'Incidentes trabalhistas', 'impacto': -incidentes_pen, 'direcao': 'negativo'})
    noticias_s_pen = round(min(entrada.noticias_negativas_12m * 3.5, 35), 1)
    if noticias_s_pen > 0:
        fatores.append({'fator': 'Notícias negativas (social)', 'impacto': -noticias_s_pen, 'direcao': 'negativo'})

    # Governança
    if entrada.possui_politica_privacidade_dados:
        fatores.append({'fator': 'Política de privacidade de dados', 'impacto': 30, 'direcao': 'positivo'})
    if entrada.possui_politica_anticorrupcao:
        fatores.append({'fator': 'Política anticorrupção', 'impacto': 30, 'direcao': 'positivo'})
    noticias_g_pen = round(min(entrada.noticias_negativas_12m * 2.0, 20), 1)
    if noticias_g_pen > 0:
        fatores.append({'fator': 'Notícias negativas (governança)', 'impacto': -noticias_g_pen, 'direcao': 'negativo'})
    cert_g = round(min(entrada.quantidade_certificacoes * 2.0, 20), 1)
    if cert_g > 0:
        fatores.append({'fator': 'Certificações (governança)', 'impacto': cert_g, 'direcao': 'positivo'})
    if entrada.consta_lista_sancoes:
        fatores.append({'fator': 'Consta em lista de sanções', 'impacto': -50, 'direcao': 'negativo'})

    fatores.sort(key=lambda f: abs(f['impacto']), reverse=True)
    return fatores


# ── Classificação individual ──────────────────────────────────────────────────

@roteador.post('/classificar', response_model=AvaliacaoSaida,
               dependencies=[Depends(obter_usuario_atual)])
def classificar(entrada: FornecedorEntrada,
                sessao: Session = Depends(obter_sessao)) -> AvaliacaoSaida:
    """Classifica um fornecedor e persiste resultado + plano de ação no banco."""
    servico = ServicoAvaliacao(repositorio)
    try:
        resultado = _classificar_e_persistir(sessao, entrada, servico)
        sessao.commit()
        return resultado
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@roteador.post('/explicabilidade',
               dependencies=[Depends(obter_usuario_atual)])
def explicabilidade(entrada: FornecedorEntrada) -> dict:
    """Retorna a classificação ESG e os fatores que influenciaram a decisão."""
    ausentes = [n for n in ('modelo_knn', 'modelo_rf', 'config') if not repositorio.existe(n)]
    if ausentes:
        raise HTTPException(
            status_code=503,
            detail=f"Modelos ausentes: {ausentes}. Execute: POST /treinar",
        )
    try:
        servico = ServicoAvaliacao(repositorio)
        d = servico.avaliar_um(_dominio(entrada))
        avaliacao = _saida(d, entrada)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        'avaliacao': avaliacao.model_dump(),
        'fatores': _fatores_explicabilidade(entrada),
    }


# ── Classificação em lote ─────────────────────────────────────────────────────

@roteador.post('/classificar/lote', response_model=ResultadoClassificacaoLote)
def classificar_lote(
    entrada: ClassificacaoLoteEntrada,
    sessao: Session = Depends(obter_sessao),
    usuario: UsuarioBanco = Depends(obter_usuario_atual),
) -> ResultadoClassificacaoLote:
    """Classifica lista de fornecedores com rastreamento de erros por linha."""
    import uuid
    servico           = ServicoAvaliacao(repositorio)
    resultados, erros = [], []
    lote_id           = str(uuid.uuid4())

    ausentes = [n for n in ('modelo_knn', 'modelo_rf', 'config') if not repositorio.existe(n)]
    if ausentes:
        raise HTTPException(
            status_code=503,
            detail=f"Modelos ausentes: {ausentes}. Execute: POST /treinar",
        )

    for idx, forn in enumerate(entrada.fornecedores, 1):
        try:
            # SAVEPOINT por linha: falha de uma linha não quebra a sessão das demais
            with sessao.begin_nested():
                resultado = _classificar_e_persistir(
                    sessao, forn, servico,
                    usuario_id=usuario.id, usuario_email=usuario.email,
                    lote_id=lote_id, operacao='lote',
                )
            resultados.append(resultado)
        except Exception as exc:
            erros.append({
                'linha':             str(idx),
                'codigo_fornecedor': forn.codigo_fornecedor,
                'razao_social':      forn.razao_social,
                'erro':              str(exc),
            })

    sessao.commit()
    return ResultadoClassificacaoLote(
        total_processados=len(resultados),
        total_erros=len(erros),
        resultados=resultados,
        erros=erros,
    )


# ── Upload de arquivo ─────────────────────────────────────────────────────────

@roteador.post('/avaliar/upload', response_model=list[AvaliacaoSaida])
async def avaliar_upload(
    arquivo: UploadFile = File(...),
    sessao: Session = Depends(obter_sessao),
    usuario: UsuarioBanco = Depends(obter_usuario_atual),
) -> list[AvaliacaoSaida]:
    """Upload CSV/XLSX com 17 colunas PT — classifica todos os fornecedores do arquivo.

    Colunas obrigatórias: codigo_fornecedor, razao_social, cnpj, setor, pais,
        possui_politica_ambiental, emissoes_carbono_ton, percentual_energia_renovavel,
        percentual_reciclagem_residuos, incidentes_trabalhistas_12m,
        possui_programa_diversidade, possui_politica_privacidade_dados,
        possui_politica_anticorrupcao, consta_lista_sancoes, noticias_negativas_12m,
        quantidade_certificacoes, receita_anual
    """
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    sufixo = Path(arquivo.filename or 'arquivo.csv').suffix or '.csv'
    with NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
        tmp.write(await arquivo.read())
        tmp_path = Path(tmp.name)

    def _safe_float(v: object, default: float = 0.0) -> float:
        try:
            r = float(v)  # type: ignore[arg-type]
            return default if pd.isna(r) else r
        except (TypeError, ValueError):
            return default

    def _safe_int(v: object, default: int = 0) -> int:
        try:
            r = float(v)  # type: ignore[arg-type]
            return default if pd.isna(r) else int(r)
        except (TypeError, ValueError):
            return default

    ausentes = [n for n in ('modelo_knn', 'modelo_rf', 'config') if not repositorio.existe(n)]
    if ausentes:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"Modelos ausentes: {ausentes}. Execute: POST /treinar",
        )

    try:
        df = (pd.read_csv(tmp_path)
              if sufixo.lower() == '.csv'
              else pd.read_excel(tmp_path))
        df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_')
                      for c in df.columns]

        def _conv_bool(v: object) -> bool:
            if isinstance(v, bool):
                return v
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {'1', 'true', 'sim', 's', 'yes', 'y'}

        for col in ['possui_politica_ambiental', 'possui_programa_diversidade',
                    'possui_politica_privacidade_dados', 'possui_politica_anticorrupcao',
                    'consta_lista_sancoes']:
            if col in df.columns:
                df[col] = df[col].map(_conv_bool)

        import uuid
        servico    = ServicoAvaliacao(repositorio)
        resultados: list[AvaliacaoSaida] = []
        lote_id    = str(uuid.uuid4())

        for _, row in df.iterrows():
            forn = FornecedorEntrada(
                codigo_fornecedor              = str(row.get('codigo_fornecedor', '') or ''),
                razao_social                   = str(row.get('razao_social', '') or ''),
                cnpj                           = str(row.get('cnpj', '') or ''),
                setor                          = str(row.get('setor', '') or ''),
                pais                           = str(row.get('pais', 'BR') or 'BR'),
                possui_politica_ambiental      = bool(row.get('possui_politica_ambiental', False)),
                emissoes_carbono_ton           = _safe_float(row.get('emissoes_carbono_ton', 0)),
                percentual_energia_renovavel   = _safe_float(row.get('percentual_energia_renovavel', 0)),
                percentual_reciclagem_residuos = _safe_float(row.get('percentual_reciclagem_residuos', 0)),
                incidentes_trabalhistas_12m    = _safe_int(row.get('incidentes_trabalhistas_12m', 0)),
                possui_programa_diversidade    = bool(row.get('possui_programa_diversidade', False)),
                possui_politica_privacidade_dados = bool(row.get('possui_politica_privacidade_dados', False)),
                possui_politica_anticorrupcao  = bool(row.get('possui_politica_anticorrupcao', False)),
                consta_lista_sancoes           = bool(row.get('consta_lista_sancoes', False)),
                noticias_negativas_12m         = _safe_int(row.get('noticias_negativas_12m', 0)),
                quantidade_certificacoes       = _safe_int(row.get('quantidade_certificacoes', 0)),
                receita_anual                  = _safe_float(row.get('receita_anual', 0)),
            )
            # SAVEPOINT por linha: erro de DB não contamina a sessão das linhas seguintes
            with sessao.begin_nested():
                resultados.append(_classificar_e_persistir(
                    sessao, forn, servico,
                    usuario_id=usuario.id, usuario_email=usuario.email,
                    lote_id=lote_id, operacao='upload',
                ))

        sessao.commit()
        return resultados

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Consulta de fornecedores ──────────────────────────────────────────────────

@roteador.get('/fornecedores', response_model=list[FornecedorSaida],
              dependencies=[Depends(obter_usuario_atual)])
def listar_fornecedores(sessao: Session = Depends(obter_sessao)) -> list[FornecedorSaida]:
    """Lista todos os fornecedores com dados cadastrais e total de avaliações."""
    registros = sessao.query(FornecedorBanco).order_by(FornecedorBanco.name).all()
    result = []
    for r in registros:
        total = sessao.query(AvaliacaoBanco).filter(
            AvaliacaoBanco.fornecedor_id == r.id).count()
        result.append(FornecedorSaida(
            id=r.id,
            name=r.name,
            industry=r.industry,
            environment_score=r.environment_score,
            social_score=r.social_score,
            governance_score=r.governance_score,
            cnpj=r.cnpj,
            email=r.email,
            telefone=r.telefone,
            contato=r.contato,
            quantidade_funcionarios=r.quantidade_funcionarios,
            endereco=r.endereco,
            website=r.website,
            descricao=r.descricao,
            criado_em=r.criado_em,
            atualizado_em=r.atualizado_em,
            total_avaliacoes=total,
        ))
    return result


# ── Histórico de avaliações por fornecedor ────────────────────────────────────

@roteador.get('/fornecedores/{fornecedor_id}/historico',
              response_model=HistoricoFornecedorSaida,
              dependencies=[Depends(obter_usuario_atual)])
def historico_fornecedor(fornecedor_id: int,
                         sessao: Session = Depends(obter_sessao)) -> HistoricoFornecedorSaida:
    """Histórico completo de avaliações ESG de um fornecedor (ordem cronológica)."""
    forn = sessao.query(FornecedorBanco).filter(FornecedorBanco.id == fornecedor_id).first()
    if not forn:
        raise HTTPException(status_code=404, detail='Fornecedor não encontrado')

    avaliacoes = (sessao.query(AvaliacaoBanco)
                  .filter(AvaliacaoBanco.fornecedor_id == fornecedor_id)
                  .order_by(AvaliacaoBanco.criado_em.desc())
                  .all())

    itens = []
    for av in avaliacoes:
        planos = (sessao.query(PlanoAcaoBanco)
                  .filter(PlanoAcaoBanco.avaliacao_id == av.id)
                  .order_by(PlanoAcaoBanco.importancia.desc())
                  .all())
        itens.append(HistoricoItemSaida(
            avaliacao_id=av.id,
            criado_em=av.criado_em,
            environment_score=av.environment_score,
            social_score=av.social_score,
            governance_score=av.governance_score,
            total_score=av.total_score,
            score_ponderado=av.score_ponderado,
            grade=av.grade,
            maturidade_rf=av.maturidade_rf,
            maturidade_knn=av.maturidade_knn,
            confianca_rf_high=av.confianca_rf_high,
            risco=av.risco,
            impacto=av.impacto,
            quadrante=av.quadrante,
            plano_acao=[
                PlanoAcaoSaida(pilar=p.pilar, score=p.score,
                               importancia=p.importancia, acao=p.acao)
                for p in planos
            ],
        ))

    return HistoricoFornecedorSaida(
        fornecedor_id=forn.id,
        name=forn.name,
        industry=forn.industry,
        total_avaliacoes=len(itens),
        avaliacoes=itens,
    )


# ── Plano de ação atual do fornecedor ─────────────────────────────────────────

@roteador.get('/fornecedores/{fornecedor_id}/plano-acao',
              response_model=list[PlanoAcaoSaida],
              dependencies=[Depends(obter_usuario_atual)])
def plano_acao_fornecedor(fornecedor_id: int,
                           sessao: Session = Depends(obter_sessao)) -> list[PlanoAcaoSaida]:
    """Plano de ação da avaliação mais recente do fornecedor."""
    ultima_av = (sessao.query(AvaliacaoBanco)
                 .filter(AvaliacaoBanco.fornecedor_id == fornecedor_id)
                 .order_by(AvaliacaoBanco.criado_em.desc())
                 .first())
    if not ultima_av:
        raise HTTPException(status_code=404,
                            detail='Fornecedor não encontrado ou sem avaliações')

    planos = (sessao.query(PlanoAcaoBanco)
              .filter(PlanoAcaoBanco.avaliacao_id == ultima_av.id)
              .order_by(PlanoAcaoBanco.importancia.desc())
              .all())
    return [
        PlanoAcaoSaida(pilar=p.pilar, score=p.score,
                       importancia=p.importancia, acao=p.acao)
        for p in planos
    ]


# ── Plano de ação de uma avaliação específica ─────────────────────────────────

@roteador.get('/avaliacoes/{avaliacao_id}/plano-acao',
              response_model=list[PlanoAcaoSaida],
              dependencies=[Depends(obter_usuario_atual)])
def plano_acao_avaliacao(avaliacao_id: int,
                          sessao: Session = Depends(obter_sessao)) -> list[PlanoAcaoSaida]:
    """Plano de ação de uma avaliação específica (por ID)."""
    av = sessao.query(AvaliacaoBanco).filter(AvaliacaoBanco.id == avaliacao_id).first()
    if not av:
        raise HTTPException(status_code=404, detail='Avaliação não encontrada')

    planos = (sessao.query(PlanoAcaoBanco)
              .filter(PlanoAcaoBanco.avaliacao_id == avaliacao_id)
              .order_by(PlanoAcaoBanco.importancia.desc())
              .all())
    return [
        PlanoAcaoSaida(pilar=p.pilar, score=p.score,
                       importancia=p.importancia, acao=p.acao)
        for p in planos
    ]


# ── Helpers de dashboard ──────────────────────────────────────────────────────

def _consultar_view_classificacoes(sessao: Session) -> list[dict]:
    """Retorna os dados de vw_fornecedores_classificacoes como lista de dicts JSON-safe."""
    from decimal import Decimal
    from datetime import datetime as _dt
    from sqlalchemy import text

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, Decimal):
            return float(v)
        if isinstance(v, _dt):
            return v.isoformat()
        return v

    try:
        rows = sessao.execute(
            text('SELECT * FROM vw_fornecedores_classificacoes')
        ).mappings().all()
        return [{k: _safe(v) for k, v in dict(r).items()} for r in rows]
    except Exception:
        return []


# ── Dashboards ────────────────────────────────────────────────────────────────

@roteador.get('/dashboard/executivo', dependencies=[Depends(obter_usuario_atual)])
def dashboard_executivo(sessao: Session = Depends(obter_sessao)) -> dict:
    """KPIs executivos — usa a avaliação mais recente por fornecedor."""
    from sqlalchemy import func

    subq = (sessao.query(func.max(AvaliacaoBanco.id).label('max_id'))
            .filter(AvaliacaoBanco.fornecedor_id.isnot(None))
            .group_by(AvaliacaoBanco.fornecedor_id)
            .subquery())

    registros = (
        sessao.query(AvaliacaoBanco).join(subq, AvaliacaoBanco.id == subq.c.max_id).all()
        + sessao.query(AvaliacaoBanco).filter(AvaliacaoBanco.fornecedor_id.is_(None)).all()
    )

    if not registros:
        return {'kpis': {}, 'distribuicao_risco': [], 'medias_pilares': [],
                'top_risco': [], 'melhores': [], 'fornecedores_classificacoes': []}

    df = pd.DataFrame([{c.name: getattr(r, c.name)
                         for c in r.__table__.columns} for r in registros])

    df['nivel_risco'] = df['risco'].apply(_nivel_risco)

    top_df = df.sort_values('risco', ascending=False).head(10).copy()
    top_df['razao_social']               = top_df['name']
    top_df['probabilidade_ml_alto_risco'] = (top_df['confianca_rf_high'] / 100.0).round(3)

    mel_df = df.sort_values('score_ponderado', ascending=False).head(10).copy()
    mel_df['razao_social'] = mel_df['name']
    mel_df['pontuacao_esg'] = (mel_df['score_ponderado'] / 10.0).round(1)

    return {
        'kpis': {
            'total_fornecedores':     len(df),
            'score_medio':            round(float(df['score_ponderado'].mean()) / 10.0, 2),
            'alto_risco':             int((df['nivel_risco'] == 'alto').sum()),
            'probabilidade_ml_media': round(float(df['confianca_rf_high'].mean()), 1),
        },
        'distribuicao_risco': (
            df.groupby('nivel_risco', as_index=False)
              .size().rename(columns={'size': 'quantidade'}).to_dict('records')
        ),
        'medias_pilares': [
            {'pilar': 'Ambiental',  'valor': round(float(df['environment_score'].mean()) / 10.0, 1)},
            {'pilar': 'Social',     'valor': round(float(df['social_score'].mean())       / 10.0, 1)},
            {'pilar': 'Governança', 'valor': round(float(df['governance_score'].mean())   / 10.0, 1)},
        ],
        'top_risco': (
            top_df[['razao_social', 'industry', 'risco',
                    'probabilidade_ml_alto_risco', 'quadrante', 'maturidade_rf']]
            .to_dict('records')
        ),
        'melhores': (
            mel_df[['razao_social', 'industry', 'pontuacao_esg', 'grade', 'maturidade_rf']]
            .to_dict('records')
        ),
        'fornecedores_classificacoes': _consultar_view_classificacoes(sessao),
    }


@roteador.get('/dashboard/ml',
              dependencies=[Depends(exigir_perfil('administrador', 'cientista_dados'))])
def dashboard_ml(sessao: Session = Depends(obter_sessao)) -> dict:
    """Dashboard de ML: métricas do modelo, dispersão e importância de features."""

    def _pct(v) -> float | None:
        return round(float(v) * 100, 2) if v is not None else None

    metricas: dict   = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    experimento: dict | None = None

    ultimo_exp = (sessao.query(ExperimentoMLBanco)
                  .order_by(ExperimentoMLBanco.criado_em.desc()).first())

    if ultimo_exp:
        metricas = {
            'accuracy':  _pct(ultimo_exp.rf_acuracia),
            'precision': _pct(ultimo_exp.rf_precision if ultimo_exp.rf_precision is not None
                              else ultimo_exp.rf_acuracia),
            'recall':    _pct(ultimo_exp.rf_recall    if ultimo_exp.rf_recall    is not None
                              else ultimo_exp.rf_acuracia),
            'f1':        _pct(ultimo_exp.rf_f1_medium),
        }
        experimento = {
            'nome':      ultimo_exp.nome_execucao,
            'criado_em': ultimo_exp.criado_em.isoformat() if ultimo_exp.criado_em else None,
            'rf': {
                'acuracia':  _pct(ultimo_exp.rf_acuracia),
                'precision': _pct(ultimo_exp.rf_precision),
                'recall':    _pct(ultimo_exp.rf_recall),
                'f1':        _pct(ultimo_exp.rf_f1_medium),
            },
            'knn': {
                'acuracia':  _pct(ultimo_exp.knn_acuracia),
                'precision': _pct(ultimo_exp.knn_precision),
                'recall':    _pct(ultimo_exp.knn_recall),
                'f1':        _pct(ultimo_exp.knn_f1_medium),
            },
        }

    # Feature importance
    feature_importance: list = []
    try:
        _, rf_meta = repositorio.carregar('modelo_rf')
        importancias = dict(rf_meta.get('importancias', []))
        _LABELS_FI = {
            'environment_score': 'Ambiental',
            'social_score':      'Social',
            'governance_score':  'Governança',
            'industry_enc':      'Setor',
        }
        feature_importance = sorted(
            [{'variavel': _LABELS_FI.get(k, k), 'importancia': round(float(v) * 100, 1)}
             for k, v in importancias.items()],
            key=lambda x: -x['importancia'],
        )
    except Exception:
        pass

    registros = sessao.query(AvaliacaoBanco).all()
    if not registros:
        return {'metricas': metricas, 'experimento': experimento,
                'distribuicao_scores': [], 'dispersao': [],
                'feature_importance': feature_importance}

    df = pd.DataFrame([{c.name: getattr(r, c.name)
                         for c in r.__table__.columns} for r in registros])

    df['nivel_risco']               = df['risco'].apply(_nivel_risco)
    df['pontuacao_esg']             = (df['score_ponderado'] / 10.0).round(1)
    df['probabilidade_ml_alto_risco'] = (df['confianca_rf_high'] / 100.0).round(3)
    df['pontuacao_ambiental']       = (df['environment_score'] / 10).astype(int)
    df['pontuacao_social']          = (df['social_score']      / 10).astype(int)
    df['pontuacao_governanca']      = (df['governance_score']  / 10).astype(int)
    df['razao_social']              = df['name']

    return {
        'metricas': metricas,
        'experimento': experimento,
        'distribuicao_scores': (
            df[['razao_social', 'pontuacao_esg', 'nivel_risco']].to_dict('records')
        ),
        'dispersao': (
            df[['razao_social', 'pontuacao_esg', 'probabilidade_ml_alto_risco',
                'nivel_risco', 'pontuacao_ambiental', 'pontuacao_social', 'pontuacao_governanca']]
            .to_dict('records')
        ),
        'feature_importance': feature_importance,
    }


# ── /dashboard/estatistico ────────────────────────────────────────────────────

@roteador.get('/dashboard/estatistico', dependencies=[Depends(obter_usuario_atual)])
def dashboard_estatistico(
    data_inicio: str | None = None,
    data_fim:    str | None = None,
    setor:       str | None = None,
    maturidade:  str | None = None,
    sessao: Session = Depends(obter_sessao),
) -> dict:
    """Estatísticas descritivas, distribuições, correlações e tendências dos dados do banco."""
    import math

    todos_setores = sorted({
        r[0] for r in sessao.query(AvaliacaoBanco.industry).distinct().all()
    })

    query = sessao.query(AvaliacaoBanco)
    if data_inicio:
        query = query.filter(AvaliacaoBanco.criado_em >= data_inicio)
    if data_fim:
        query = query.filter(AvaliacaoBanco.criado_em <= data_fim + ' 23:59:59')
    if setor:
        query = query.filter(AvaliacaoBanco.industry == setor)
    if maturidade:
        query = query.filter(AvaliacaoBanco.maturidade_rf == maturidade)

    registros = query.all()

    _vazio = {
        'kpis': {}, 'estatisticas': {}, 'distribuicao_risco': [],
        'distribuicao_maturidade': [], 'distribuicao_grade': [],
        'por_setor': [], 'serie_temporal': [], 'correlacoes': {},
        'planos_por_pilar': [], 'historico_ml': [],
        'registros_detalhe': [], 'setores_disponiveis': todos_setores,
        'filtros_aplicados': {
            'data_inicio': data_inicio, 'data_fim': data_fim,
            'setor': setor, 'maturidade': maturidade,
        },
    }
    if not registros:
        return _vazio

    def _f(v) -> float | None:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), 3)

    df = pd.DataFrame([{c.name: getattr(r, c.name)
                         for c in r.__table__.columns} for r in registros])
    df['nivel_risco']   = df['risco'].apply(_nivel_risco)
    df['pontuacao_esg'] = (df['score_ponderado'] / 10.0).round(2)
    df['env']           = (df['environment_score'] / 10.0).round(2)
    df['soc']           = (df['social_score']      / 10.0).round(2)
    df['gov']           = (df['governance_score']  / 10.0).round(2)
    df['criado_em']     = pd.to_datetime(df['criado_em'], utc=True)
    df['periodo']       = df['criado_em'].dt.to_period('M').astype(str)

    total = len(df)

    def _stats(s: pd.Series) -> dict:
        s = s.dropna()
        if s.empty:
            return {}
        moda_s = s.mode()
        return {
            'media':         _f(s.mean()),
            'mediana':       _f(s.median()),
            'moda':          _f(moda_s.iloc[0]) if not moda_s.empty else None,
            'variancia':     _f(s.var()),
            'desvio_padrao': _f(s.std()),
            'minimo':        _f(s.min()),
            'maximo':        _f(s.max()),
            'q1':            _f(s.quantile(0.25)),
            'q3':            _f(s.quantile(0.75)),
            'total':         int(len(s)),
        }

    kpis = {
        'total_avaliacoes':           total,
        'total_fornecedores':         int(df['fornecedor_id'].nunique()),
        'total_setores':              int(df['industry'].nunique()),
        'score_medio':                _f(df['pontuacao_esg'].mean()),
        'risco_medio':                _f(df['risco'].mean()),
        'percentual_alto_risco':      _f((df['nivel_risco'] == 'alto').sum() / total * 100),
        'percentual_maturidade_high': _f((df['maturidade_rf'] == 'High').sum() / total * 100),
    }

    estatisticas = {
        'ambiental':  _stats(df['env']),
        'social':     _stats(df['soc']),
        'governanca': _stats(df['gov']),
        'total_esg':  _stats(df['pontuacao_esg']),
        'risco':      _stats(df['risco']),
    }

    _dr = df.groupby('nivel_risco', as_index=False).size().rename(columns={'size': 'quantidade'})
    _dr['percentual'] = (_dr['quantidade'] / total * 100).round(1)
    distribuicao_risco = _dr.to_dict('records')

    _dm = df.groupby('maturidade_rf', as_index=False).size().rename(columns={'size': 'quantidade'})
    _dm['percentual'] = (_dm['quantidade'] / total * 100).round(1)
    distribuicao_maturidade = _dm.to_dict('records')

    distribuicao_grade = (
        df.groupby('grade', as_index=False).size()
          .rename(columns={'size': 'quantidade'})
          .sort_values('grade')
          .to_dict('records')
    )

    por_setor = (
        df.groupby('industry')
          .agg(total=('id', 'count'),
               score_medio=('pontuacao_esg', 'mean'),
               risco_medio=('risco', 'mean'),
               desvio_padrao=('pontuacao_esg', 'std'))
          .round(2).reset_index()
          .rename(columns={'industry': 'setor'})
          .sort_values('score_medio', ascending=False)
          .to_dict('records')
    )

    serie_temporal = (
        df.groupby('periodo')
          .agg(total=('id', 'count'),
               score_medio=('pontuacao_esg', 'mean'),
               risco_medio=('risco', 'mean'))
          .round(2).reset_index()
          .sort_values('periodo')
          .to_dict('records')
    )

    _cols_corr = {
        'env': 'Ambiental', 'soc': 'Social', 'gov': 'Governança',
        'pontuacao_esg': 'Score ESG', 'risco': 'Risco', 'impacto': 'Impacto',
    }
    _cm = df[list(_cols_corr.keys())].corr().round(3)
    _cm.index   = list(_cols_corr.values())
    _cm.columns = list(_cols_corr.values())
    correlacoes = {
        'labels': list(_cols_corr.values()),
        'matrix': [[_f(v) for v in row] for row in _cm.values.tolist()],
    }

    ids_avaliacao = set(int(i) for i in df['id'].tolist())
    planos_registros = (
        sessao.query(PlanoAcaoBanco)
              .filter(PlanoAcaoBanco.avaliacao_id.in_(ids_avaliacao))
              .all()
    ) if ids_avaliacao else []
    planos_por_pilar: list = []
    if planos_registros:
        df_p = pd.DataFrame([{c.name: getattr(r, c.name)
                               for c in r.__table__.columns} for r in planos_registros])
        planos_por_pilar = (
            df_p.groupby('pilar')
                .agg(total=('id', 'count'),
                     score_medio=('score', 'mean'),
                     importancia_media=('importancia', 'mean'))
                .round(3).reset_index()
                .to_dict('records')
        )

    experimentos = (sessao.query(ExperimentoMLBanco)
                    .order_by(ExperimentoMLBanco.criado_em.asc()).all())
    historico_ml = [
        {
            'nome':        e.nome_execucao,
            'criado_em':   e.criado_em.isoformat() if e.criado_em else None,
            'rf_acuracia': round(float(e.rf_acuracia or 0) * 100, 2),
            'knn_acuracia': round(float(e.knn_acuracia or 0) * 100, 2),
            'rf_f1':       round(float(e.rf_f1_medium or 0) * 100, 2),
            'knn_f1':      round(float(e.knn_f1_medium or 0) * 100, 2),
        }
        for e in experimentos
    ]

    registros_detalhe = (
        df[['industry', 'env', 'soc', 'gov', 'pontuacao_esg',
            'risco', 'impacto', 'nivel_risco', 'maturidade_rf', 'grade', 'periodo']]
        .rename(columns={
            'industry': 'setor', 'env': 'ambiental',
            'soc': 'social', 'gov': 'governanca',
        })
        .to_dict('records')
    )

    return {
        'kpis':                    kpis,
        'estatisticas':            estatisticas,
        'distribuicao_risco':      distribuicao_risco,
        'distribuicao_maturidade': distribuicao_maturidade,
        'distribuicao_grade':      distribuicao_grade,
        'por_setor':               por_setor,
        'serie_temporal':          serie_temporal,
        'correlacoes':             correlacoes,
        'planos_por_pilar':        planos_por_pilar,
        'historico_ml':            historico_ml,
        'registros_detalhe':       registros_detalhe,
        'setores_disponiveis':     todos_setores,
        'filtros_aplicados': {
            'data_inicio': data_inicio, 'data_fim': data_fim,
            'setor': setor, 'maturidade': maturidade,
        },
    }
