# esg_ml/dominio/entidades/empresa.py
# CRISP-DM Fase 1 — Vocabulário do negócio ESG (Edenred Brasil)
# Zero dependências externas — Python puro

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ScoreESG:
    """Score ESG imutável por empresa. Escala 0–1000 por pilar."""
    environment_score: int
    social_score:      int
    governance_score:  int

    def __post_init__(self):
        for campo, v in [('environment_score', self.environment_score),
                         ('social_score',       self.social_score),
                         ('governance_score',   self.governance_score)]:
            if not (0 <= v <= 1000):
                raise ValueError(f'{campo} deve ser 0–1000. Recebido: {v}')

    @property
    def total(self) -> int:
        return self.environment_score + self.social_score + self.governance_score


@dataclass(frozen=True)
class PesosSetor:
    """Pesos de cada pilar para um setor (soma = 1.0)."""
    w_E:   float
    w_S:   float
    w_G:   float
    fonte: str

    def __post_init__(self):
        if abs(self.w_E + self.w_S + self.w_G - 1.0) > 0.01:
            raise ValueError('Pesos devem somar 1.0')


class Maturidade:
    """Níveis de maturidade ESG — vocabulário do negócio."""
    AVANCADO  = 'Avançado'
    INICIANTE = 'Iniciante'
    MAPA      = {'High': AVANCADO, 'Medium': INICIANTE}

    @classmethod
    def de_nivel(cls, nivel: str) -> str:
        return cls.MAPA.get(nivel, nivel)


class Quadrante:
    """Quadrantes da Matriz de Criticidade (corte em 50)."""
    ACAO_IMEDIATA = 'Alto Impacto / Alto Risco'
    ENGAJAMENTO   = 'Alto Impacto / Baixo Risco'
    MONITORAR     = 'Baixo Impacto / Alto Risco'
    BAIXO_RISCO   = 'Baixo Impacto / Baixo Risco'

    @classmethod
    def classificar(cls, impacto: float, risco: float) -> str:
        hi = impacto > 50
        hr = risco   > 50
        if   hi and hr: return cls.ACAO_IMEDIATA
        elif hi:        return cls.ENGAJAMENTO
        elif hr:        return cls.MONITORAR
        else:           return cls.BAIXO_RISCO


@dataclass
class Empresa:
    """Entidade central — empresa com identidade, scores ESG e dados cadastrais opcionais.

    Campos obrigatórios para a engine de ML: name, industry, scores (E/S/G).
    Campos opcionais são usados apenas para cadastro/identificação no banco de dados
    e não influenciam a classificação.
    """
    # ── Obrigatórios (engine ML) ──────────────────────────────────────────
    name:     str
    industry: str
    scores:   ScoreESG

    # ── Opcionais (cadastro / identificação) ──────────────────────────────
    cnpj:                    Optional[str] = field(default=None)
    email:                   Optional[str] = field(default=None)
    telefone:                Optional[str] = field(default=None)
    contato:                 Optional[str] = field(default=None)  # nome do responsável
    quantidade_funcionarios: Optional[int] = field(default=None)
    endereco:                Optional[str] = field(default=None)
    website:                 Optional[str] = field(default=None)
    descricao:               Optional[str] = field(default=None)


PLANOS_ACAO = {
    'E': [
        'Implementar processo formal de gestão de impactos ambientais (PGRSS/PGRS).',
        'Realizar levantamento e estimativa da pegada de carbono (GHG Protocol).',
        'Buscar certificação ISO 14001 ou equivalente de gestão ambiental.',
        'Implantar programa de eficiência energética e gestão de resíduos.',
        'Contratar auditoria externa do inventário de Gases de Efeito Estufa.',
    ],
    'S': [
        'Formalizar política de compromisso com trabalho digno e direitos humanos.',
        'Implementar programa estruturado de diversidade, equidade e inclusão.',
        'Criar programa de saúde mental e bem-estar para colaboradores.',
        'Desenvolver ações de engajamento e voluntariado com a comunidade.',
        'Realizar treinamentos regulares de sustentabilidade para colaboradores.',
    ],
    'G': [
        'Aprovar e publicar política formal de responsabilidade socioambiental.',
        'Implantar código de conduta anticorrupção e canal de denúncias.',
        'Incluir cláusulas ESG nos contratos com fornecedores.',
        'Definir critérios socioambientais para seleção de fornecedores.',
        'Realizar auditorias ESG periódicas na cadeia de suprimentos.',
    ],
}

LIMIAR_CONFORMIDADE = 400  # score < 400 → pilar entra no plano de ação
