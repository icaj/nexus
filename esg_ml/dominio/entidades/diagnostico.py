# esg_ml/dominio/entidades/diagnostico.py
# CRISP-DM Fase 6 — Objeto de resultado completo do diagnóstico ESG

from dataclasses import dataclass, field
from typing import List


@dataclass
class ItemPlanoAcao:
    pilar:       str
    score:       int
    importancia: float
    acao:        str


@dataclass
class DiagnosticoESG:
    """Resultado completo do diagnóstico ESG de um fornecedor."""
    name:     str
    industry: str
    environment_score: int
    social_score:      int
    governance_score:  int
    total_score:       int
    score_ponderado:   float
    w_E:               float
    w_S:               float
    w_G:               float
    fonte_pesos:       str
    grade:             str
    level:             str
    risco:             float
    impacto:           float
    quadrante:         str
    maturidade_rf:         str
    confianca_rf_high:     float
    confianca_rf_medium:   float
    maturidade_knn:        str
    confianca_knn_high:    float
    confianca_knn_medium:  float
    plano_acao: List[ItemPlanoAcao] = field(default_factory=list)

    @property
    def modelos_concordam(self) -> bool:
        return self.maturidade_rf == self.maturidade_knn

    def to_dict(self) -> dict:
        acoes = ' | '.join(i.acao for i in self.plano_acao) \
                if self.plano_acao else 'Todos os pilares acima do limiar'
        return {
            'name': self.name, 'industry': self.industry,
            'environment_score': self.environment_score,
            'social_score': self.social_score,
            'governance_score': self.governance_score,
            'total_score': self.total_score,
            'score_ponderado': self.score_ponderado,
            'grade': self.grade, 'level': self.level,
            'w_E': round(self.w_E, 4), 'w_S': round(self.w_S, 4),
            'w_G': round(self.w_G, 4), 'fonte_pesos': self.fonte_pesos,
            'risco': self.risco, 'impacto': self.impacto,
            'quadrante': self.quadrante,
            'maturidade_rf': self.maturidade_rf,
            'confianca_rf_high': self.confianca_rf_high,
            'confianca_rf_medium': self.confianca_rf_medium,
            'maturidade_knn': self.maturidade_knn,
            'confianca_knn_high': self.confianca_knn_high,
            'confianca_knn_medium': self.confianca_knn_medium,
            'acoes_recomendadas': acoes,
        }
