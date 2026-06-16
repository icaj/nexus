# esg_ml/infraestrutura/modelos_banco.py
# ORM SQLAlchemy — tabelas para ESG Nexus Enterprise
#
# Diagrama de relacionamentos:
#   FornecedorBanco (1) ──< AvaliacaoBanco  (histórico de avaliações)
#   AvaliacaoBanco  (1) ──< PlanoAcaoBanco  (itens do plano de ação por avaliação)
#   FornecedorBanco (1) ──< PlanoAcaoBanco  (atalho direto para plano atual)

from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Identity, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from esg_ml.infraestrutura.banco_dados import Base


class UsuarioBanco(Base):
    __tablename__ = 'usuarios'
    id:          Mapped[int]      = mapped_column(Integer, Identity(always=False), primary_key=True, index=True)
    nome:        Mapped[str]      = mapped_column(String(120), nullable=False)
    email:       Mapped[str]      = mapped_column(String(180), nullable=False, unique=True, index=True)
    senha_hash:  Mapped[str]      = mapped_column(String(255), nullable=False)
    perfil:      Mapped[str]      = mapped_column(String(30),  nullable=False, default='analista_esg')
    ativo:       Mapped[bool]     = mapped_column(Boolean,     nullable=False, default=True)
    criado_em:   Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                   default=lambda: datetime.now(timezone.utc))


class FornecedorBanco(Base):
    """Cadastro de fornecedor.

    Campos obrigatórios (usados pela engine ML):
        name, industry, environment_score, social_score, governance_score

    Campos opcionais (identificação/cadastro — não afetam a classificação):
        cnpj, email, telefone, contato, quantidade_funcionarios, endereco, website, descricao
    """
    __tablename__ = 'fornecedores'

    # ── Identificação ─────────────────────────────────────────────────────
    id:                Mapped[int]      = mapped_column(Integer, Identity(always=False), primary_key=True, index=True)
    name:              Mapped[str]      = mapped_column(String(220), nullable=False, index=True)
    industry:          Mapped[str]      = mapped_column(String(100), nullable=False)

    # ── Scores ESG (obrigatórios para ML) ─────────────────────────────────
    environment_score: Mapped[int]      = mapped_column(Integer, nullable=False)
    social_score:      Mapped[int]      = mapped_column(Integer, nullable=False)
    governance_score:  Mapped[int]      = mapped_column(Integer, nullable=False)

    # ── Dados cadastrais opcionais ─────────────────────────────────────────
    cnpj:                    Mapped[str | None] = mapped_column(String(20),  nullable=True)
    email:                   Mapped[str | None] = mapped_column(String(180), nullable=True)
    telefone:                Mapped[str | None] = mapped_column(String(30),  nullable=True)
    contato:                 Mapped[str | None] = mapped_column(String(120), nullable=True)
    quantidade_funcionarios: Mapped[int | None] = mapped_column(Integer,     nullable=True)
    endereco:                Mapped[str | None] = mapped_column(String(255), nullable=True)
    website:                 Mapped[str | None] = mapped_column(String(255), nullable=True)
    descricao:               Mapped[str | None] = mapped_column(Text,        nullable=True)

    # ── Auditoria ─────────────────────────────────────────────────────────
    criado_em:     Mapped[datetime]      = mapped_column(DateTime(timezone=True),
                                                          default=lambda: datetime.now(timezone.utc))
    atualizado_em: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True,
                                                            onupdate=lambda: datetime.now(timezone.utc))

    # ── Relacionamentos ───────────────────────────────────────────────────
    avaliacoes:  Mapped[list["AvaliacaoBanco"]] = relationship(back_populates="fornecedor")
    
    planos_acao: list['PlanoAcaoBanco'] = relationship('PlanoAcaoBanco',  back_populates='fornecedor',
                                                        lazy='dynamic')


class AvaliacaoBanco(Base):
    """Diagnóstico ESG completo — um registro por avaliação (histórico).

    Cada chamada a /classificar ou /avaliar/upload gera um novo registro,
    permitindo rastreio temporal da evolução do fornecedor.

    Relacionamentos:
        fornecedor_id -> FornecedorBanco (FK, nullable para retrocompatibilidade)
        planos_acao   -> PlanoAcaoBanco  (itens estruturados do plano de ação)
    """
    __tablename__ = 'avaliacoes_esg'

    id:                    Mapped[int]      = mapped_column(Integer, Identity(always=False), primary_key=True, index=True)

    # ── FK para histórico por fornecedor ──────────────────────────────────
    fornecedor_id:         Mapped[int | None] = mapped_column(Integer,
                                                               ForeignKey('fornecedores.id', ondelete='SET NULL'),
                                                               nullable=True, index=True)

    # ── Identificação (desnormalizado para consultas autônomas) ───────────
    name:                  Mapped[str]      = mapped_column(String(220), nullable=False, index=True)
    industry:              Mapped[str]      = mapped_column(String(100), nullable=False)

    # ── Scores de entrada ─────────────────────────────────────────────────
    environment_score:     Mapped[int]      = mapped_column(Integer, nullable=False)
    social_score:          Mapped[int]      = mapped_column(Integer, nullable=False)
    governance_score:      Mapped[int]      = mapped_column(Integer, nullable=False)

    # ── Scores calculados ─────────────────────────────────────────────────
    total_score:           Mapped[int]      = mapped_column(Integer, nullable=False)
    score_ponderado:       Mapped[float]    = mapped_column(Float,   nullable=False)
    grade:                 Mapped[str]      = mapped_column(String(10),  nullable=False)
    level:                 Mapped[str]      = mapped_column(String(40),  nullable=False)

    # ── Pesos do setor ────────────────────────────────────────────────────
    w_E:                   Mapped[float]    = mapped_column(Float, nullable=False)
    w_S:                   Mapped[float]    = mapped_column(Float, nullable=False)
    w_G:                   Mapped[float]    = mapped_column(Float, nullable=False)
    fonte_pesos:           Mapped[str]      = mapped_column(String(80),  nullable=False)

    # ── Risco / Impacto / Quadrante ───────────────────────────────────────
    risco:                 Mapped[float]    = mapped_column(Float, nullable=False)
    impacto:               Mapped[float]    = mapped_column(Float, nullable=False)
    quadrante:             Mapped[str]      = mapped_column(String(60),  nullable=False)

    # ── Resultados ML ─────────────────────────────────────────────────────
    maturidade_rf:         Mapped[str]      = mapped_column(String(20),  nullable=False)
    confianca_rf_high:     Mapped[float]    = mapped_column(Float, nullable=False)
    confianca_rf_medium:   Mapped[float]    = mapped_column(Float, nullable=False)
    maturidade_knn:        Mapped[str]      = mapped_column(String(20),  nullable=False)
    confianca_knn_high:    Mapped[float]    = mapped_column(Float, nullable=False)
    confianca_knn_medium:  Mapped[float]    = mapped_column(Float, nullable=False)

    # ── Plano de ação (resumo textual — retrocompatibilidade) ─────────────
    acoes_recomendadas:    Mapped[str]      = mapped_column(Text,  nullable=False, default='')

    # ── Auditoria ─────────────────────────────────────────────────────────
    criado_em:             Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                             default=lambda: datetime.now(timezone.utc))

    # ── Relacionamentos ───────────────────────────────────────────────────
    fornecedor:  'FornecedorBanco'      = relationship('FornecedorBanco', back_populates='avaliacoes')
    planos_acao: list['PlanoAcaoBanco'] = relationship('PlanoAcaoBanco',  back_populates='avaliacao',
                                                        cascade='all, delete-orphan',
                                                        order_by='PlanoAcaoBanco.importancia.desc()')


class PlanoAcaoBanco(Base):
    """Item individual do plano de ação ESG gerado para uma avaliação.

    Cada avaliação pode gerar múltiplos itens (um por ação recomendada por pilar).
    Relacionado a AvaliacaoBanco (obrigatório) e FornecedorBanco (atalho direto).
    """
    __tablename__ = 'planos_acao'

    id:            Mapped[int]      = mapped_column(Integer, Identity(always=False), primary_key=True, index=True)
    avaliacao_id:  Mapped[int]      = mapped_column(Integer,
                                                     ForeignKey('avaliacoes_esg.id', ondelete='CASCADE'),
                                                     nullable=False, index=True)
    fornecedor_id: Mapped[int | None] = mapped_column(Integer,
                                                       ForeignKey('fornecedores.id', ondelete='SET NULL'),
                                                       nullable=True, index=True)

    # ── Dados do item ─────────────────────────────────────────────────────
    pilar:        Mapped[str]   = mapped_column(String(5),   nullable=False)   # 'E', 'S' ou 'G'
    score:        Mapped[int]   = mapped_column(Integer,     nullable=False)   # score do pilar
    importancia:  Mapped[float] = mapped_column(Float,       nullable=False)   # feature importance RF
    acao:         Mapped[str]   = mapped_column(Text,        nullable=False)   # texto da ação

    # ── Auditoria ─────────────────────────────────────────────────────────
    criado_em:    Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                    default=lambda: datetime.now(timezone.utc))

    # ── Relacionamentos ───────────────────────────────────────────────────
    avaliacao:  'AvaliacaoBanco'   = relationship('AvaliacaoBanco',  back_populates='planos_acao')
    fornecedor: 'FornecedorBanco'  = relationship('FornecedorBanco', back_populates='planos_acao')


class ExperimentoMLBanco(Base):
    __tablename__ = 'experimentos_ml'
    id:              Mapped[int]         = mapped_column(Integer, Identity(always=False), primary_key=True)
    nome_execucao:   Mapped[str]         = mapped_column(String(120), nullable=False)
    knn_acuracia:    Mapped[float]       = mapped_column(Float, default=0)
    knn_f1_medium:   Mapped[float]       = mapped_column(Float, default=0)
    knn_precision:   Mapped[float|None]  = mapped_column(Float, nullable=True)
    knn_recall:      Mapped[float|None]  = mapped_column(Float, nullable=True)
    rf_acuracia:     Mapped[float]       = mapped_column(Float, default=0)
    rf_f1_medium:    Mapped[float]       = mapped_column(Float, default=0)
    rf_precision:    Mapped[float|None]  = mapped_column(Float, nullable=True)
    rf_recall:       Mapped[float|None]  = mapped_column(Float, nullable=True)
    knn_params:      Mapped[str]         = mapped_column(Text, nullable=True)
    rf_params:       Mapped[str]         = mapped_column(Text, nullable=True)
    mlflow_run_id:   Mapped[str|None]    = mapped_column(String(120), nullable=True)
    criado_em:       Mapped[datetime]    = mapped_column(DateTime(timezone=True),
                                                          default=lambda: datetime.now(timezone.utc))


class LogImportacaoFornecedorBanco(Base):
    """Log de importações de fornecedores — populado via stored procedure sp_registrar_importacao."""
    __tablename__ = 'log_importacoes_fornecedores'

    id:            Mapped[int]        = mapped_column(Integer, Identity(always=False), primary_key=True)
    data_hora:     Mapped[datetime]   = mapped_column(DateTime(timezone=True),
                                                       default=lambda: datetime.now(timezone.utc))
    fornecedor_id: Mapped[int | None] = mapped_column(Integer,
                                                       ForeignKey('fornecedores.id', ondelete='SET NULL'),
                                                       nullable=True, index=True)
    razao_social:  Mapped[str | None] = mapped_column(String(220), nullable=True)
    cnpj:          Mapped[str | None] = mapped_column(String(20),  nullable=True)
    usuario_id:    Mapped[int | None] = mapped_column(Integer,
                                                       ForeignKey('usuarios.id', ondelete='SET NULL'),
                                                       nullable=True)
    usuario_email: Mapped[str | None] = mapped_column(String(180), nullable=True)
    operacao:      Mapped[str]        = mapped_column(String(20),  nullable=False, default='importacao')
    lote_id:       Mapped[str | None] = mapped_column(String(36),  nullable=True, index=True)
    detalhes:      Mapped[str | None] = mapped_column(Text, nullable=True)


class LogAlteracaoFornecedorBanco(Base):
    """Auditoria de alterações na tabela fornecedores — populado via trigger trg_audit_fornecedores."""
    __tablename__ = 'log_alteracoes_fornecedores'

    id:               Mapped[int]        = mapped_column(Integer, Identity(always=False), primary_key=True)
    data_hora:        Mapped[datetime]   = mapped_column(DateTime(timezone=True),
                                                          default=lambda: datetime.now(timezone.utc))
    fornecedor_id:    Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    operacao:         Mapped[str]        = mapped_column(String(10), nullable=False)
    dados_anteriores: Mapped[str | None] = mapped_column(Text, nullable=True)
    dados_novos:      Mapped[str | None] = mapped_column(Text, nullable=True)
    usuario_bd:       Mapped[str | None] = mapped_column(String(120), nullable=True)
