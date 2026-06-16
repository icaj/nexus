# esg_ml/aplicacao/servico_avaliacao.py
# Caso de uso: Avaliar fornecedores via nexus_v2 puro (RF modelo principal)

from typing import List
from esg_ml.adaptadores.saida.repositorio_modelo_joblib import RepositorioModeloJoblib
from esg_ml.dominio.entidades.empresa import Empresa, PesosSetor
from esg_ml.dominio.entidades.diagnostico import DiagnosticoESG
from esg_ml.dominio.servicos.predicao import diagnosticar
from esg_ml.infraestrutura.configuracoes import Configuracoes
from esg_ml.infraestrutura.registro_log import obter_registrador

registrador = obter_registrador(__name__)
conf = Configuracoes()


def _carregar_artefatos(repositorio: RepositorioModeloJoblib) -> tuple:
    ausentes = [n for n in ['modelo_knn','modelo_rf','config'] if not repositorio.existe(n)]
    if ausentes:
        raise FileNotFoundError(
            f'Modelos ausentes: {ausentes}\n'
            'Execute: python treinar_modelo.py')
    knn_model, knn_meta = repositorio.carregar('modelo_knn')
    rf_model,  rf_meta  = repositorio.carregar('modelo_rf')
    _,         cfg      = repositorio.carregar('config')
    knn_pack = {'modelo': knn_model, **knn_meta}
    rf_pack  = {'modelo': rf_model,  **rf_meta}
    return knn_pack, rf_pack, cfg


def _obter_pesos(industry: str, cfg: dict) -> PesosSetor:
    pesos_por_ind = cfg.get('pesos_por_ind', {})
    pesos_global  = cfg.get('pesos_global', {
        'environment_score': 0.3865, 'social_score': 0.3267, 'governance_score': 0.2869})
    p = pesos_por_ind.get(industry)
    if isinstance(p, PesosSetor): return p
    if isinstance(p, dict):
        return PesosSetor(w_E=p['w_E'], w_S=p['w_S'], w_G=p['w_G'], fonte=p['fonte'])
    return PesosSetor(
        w_E=pesos_global['environment_score'],
        w_S=pesos_global['social_score'],
        w_G=pesos_global['governance_score'],
        fonte='fallback global')


class ServicoAvaliacao:
    """
    Avaliação via nexus_v2:
    - Classificação: apenas ML (RF modelo principal, KNN para benchmarking)
    - Sem regras determinísticas híbridas
    - Retorna DiagnosticoESG completo com 23 campos
    """

    def __init__(self, repositorio: RepositorioModeloJoblib = None) -> None:
        self.repositorio = repositorio or RepositorioModeloJoblib(conf.diretorio_artefatos)

    def avaliar(self, fornecedores: List[Empresa]) -> List[DiagnosticoESG]:
        knn_pack, rf_pack, cfg = _carregar_artefatos(self.repositorio)
        benchmark = cfg.get('benchmark', {}).get('score_ponderado', {})
        diagnosticos = []
        for empresa in fornecedores:
            pesos = _obter_pesos(empresa.industry, cfg)
            d = diagnosticar(empresa, pesos, benchmark, knn_pack, rf_pack, cfg)
            diagnosticos.append(d)
            registrador.info('empresa_avaliada',
                             nome=empresa.name,
                             maturidade_rf=d.maturidade_rf,
                             risco=d.risco)
        return diagnosticos

    def avaliar_um(self, empresa: Empresa) -> DiagnosticoESG:
        return self.avaliar([empresa])[0]
