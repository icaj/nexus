# esg_ml/interfaces/api/esquemas.py — Pydantic schemas nexus_v3
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Entrada ───────────────────────────────────────────────────────────────────

class FornecedorEntrada(BaseModel):
    """Entrada da API — 17 métricas de domínio em português."""
    codigo_fornecedor:                str
    razao_social:                     str
    cnpj:                             str
    setor:                            str
    pais:                             str   = "BR"
    possui_politica_ambiental:        bool  = False
    emissoes_carbono_ton:             float = Field(ge=0)
    percentual_energia_renovavel:     float = Field(ge=0, le=100)
    percentual_reciclagem_residuos:   float = Field(ge=0, le=100)
    incidentes_trabalhistas_12m:      int   = Field(ge=0)
    possui_programa_diversidade:      bool  = False
    possui_politica_privacidade_dados: bool = False
    possui_politica_anticorrupcao:    bool  = False
    consta_lista_sancoes:             bool  = False
    noticias_negativas_12m:           int   = Field(ge=0)
    quantidade_certificacoes:         int   = Field(ge=0)
    receita_anual:                    float = Field(ge=0)


class ClassificacaoLoteEntrada(BaseModel):
    fornecedores: list[FornecedorEntrada]


# ── Saída: Plano de Ação ──────────────────────────────────────────────────────

class PlanoAcaoSaida(BaseModel):
    pilar:       str
    score:       int
    importancia: float
    acao:        str


# ── Saída: Avaliação completa ─────────────────────────────────────────────────

class AvaliacaoSaida(BaseModel):
    """Resultado completo do diagnóstico ESG — inclui campos PT para o frontend."""

    # ── Identificação PT ──────────────────────────────────────────────────
    codigo_fornecedor:           Optional[str]   = None
    razao_social:                Optional[str]   = None
    setor:                       Optional[str]   = None

    # ── Pontuação 0-100 (escala do frontend) ──────────────────────────────
    pontuacao_ambiental:         Optional[int]   = None
    pontuacao_social:            Optional[int]   = None
    pontuacao_governanca:        Optional[int]   = None
    pontuacao_esg:               Optional[float] = None

    # ── Classificação PT ──────────────────────────────────────────────────
    nivel_risco:                 Optional[str]   = None
    recomendacao:                Optional[str]   = None
    probabilidade_ml_alto_risco: Optional[float] = None
    motivos:                     list[str]       = []

    # ── Campos ML internos (mantidos para compatibilidade) ────────────────
    name:                  str
    industry:              str
    environment_score:     int
    social_score:          int
    governance_score:      int
    total_score:           int
    score_ponderado:       float
    grade:                 str
    level:                 str
    w_E:                   float
    w_S:                   float
    w_G:                   float
    fonte_pesos:           str
    risco:                 float
    impacto:               float
    quadrante:             str
    maturidade_rf:         str
    confianca_rf_high:     float
    confianca_rf_medium:   float
    maturidade_knn:        str
    confianca_knn_high:    float
    confianca_knn_medium:  float
    acoes_recomendadas:    str
    plano_acao:            list[PlanoAcaoSaida] = []


class ResultadoClassificacaoLote(BaseModel):
    total_processados: int
    total_erros:       int
    resultados:        list[AvaliacaoSaida]
    erros:             list[dict[str, str]]


# ── Saída: Histórico ──────────────────────────────────────────────────────────

class HistoricoItemSaida(BaseModel):
    avaliacao_id:      int
    criado_em:         datetime
    environment_score: int
    social_score:      int
    governance_score:  int
    total_score:       int
    score_ponderado:   float
    grade:             str
    maturidade_rf:     str
    maturidade_knn:    str
    confianca_rf_high: float
    risco:             float
    impacto:           float
    quadrante:         str
    plano_acao:        list[PlanoAcaoSaida] = []


class HistoricoFornecedorSaida(BaseModel):
    fornecedor_id:    int
    name:             str
    industry:         str
    total_avaliacoes: int
    avaliacoes:       list[HistoricoItemSaida]


# ── Saída: Fornecedor com dados cadastrais ────────────────────────────────────

class FornecedorSaida(BaseModel):
    """Dados do fornecedor cadastrado."""
    id:                      int
    codigo_fornecedor:       Optional[str] = None
    razao_social:            Optional[str] = None
    name:                    str
    industry:                str
    environment_score:       int
    social_score:            int
    governance_score:        int
    cnpj:                    Optional[str] = None
    email:                   Optional[str] = None
    telefone:                Optional[str] = None
    contato:                 Optional[str] = None
    quantidade_funcionarios: Optional[int] = None
    endereco:                Optional[str] = None
    website:                 Optional[str] = None
    descricao:               Optional[str] = None
    criado_em:               Optional[datetime] = None
    atualizado_em:           Optional[datetime] = None
    total_avaliacoes:        int = 0
