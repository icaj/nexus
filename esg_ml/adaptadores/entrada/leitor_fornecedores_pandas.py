# esg_ml/adaptadores/entrada/leitor_fornecedores_pandas.py
# Porta: RepositorioEmpresas → CSV/Excel com colunas obrigatórias e opcionais
#
# Colunas OBRIGATÓRIAS (engine ML):
#   name, industry, environment_score, social_score, governance_score
#
# Colunas OPCIONAIS (cadastro do fornecedor):
#   cnpj, email, telefone, contato, quantidade_funcionarios, endereco, website, descricao

import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional
from esg_ml.dominio.entidades.empresa import Empresa, ScoreESG

# Colunas obrigatórias — engine de classificação ML
COLUNAS_OBRIGATORIAS = ['name', 'industry', 'environment_score', 'social_score', 'governance_score']

# Colunas opcionais — cadastro/identificação do fornecedor
COLUNAS_OPCIONAIS = [
    'cnpj',
    'email',
    'telefone',
    'contato',
    'quantidade_funcionarios',
    'endereco',
    'website',
    'descricao',
]


def _str_ou_none(valor) -> Optional[str]:
    """Converte célula para str ou None se vazia/NaN."""
    if valor is None:
        return None
    try:
        import math
        if math.isnan(float(valor)):
            return None
    except (TypeError, ValueError):
        pass
    s = str(valor).strip()
    return s if s else None


def _int_ou_none(valor) -> Optional[int]:
    """Converte célula para int ou None se vazia/inválida."""
    if valor is None:
        return None
    try:
        import math
        f = float(valor)
        if math.isnan(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


class LeitorFornecedoresPandas:
    """Lê CSV ou XLSX e retorna lista de Empresa.

    Valida a presença de todas as colunas obrigatórias e ignora
    silenciosamente as colunas opcionais ausentes.
    """

    def __init__(self, caminho):
        self.caminho = Path(caminho)

    def ler(self) -> List[Empresa]:
        return self.listar_empresas()

    def listar_empresas(self) -> List[Empresa]:
        try:
            df = (pd.read_excel(self.caminho)
                  if self.caminho.suffix.lower() in {'.xlsx', '.xls'}
                  else pd.read_csv(self.caminho))
        except FileNotFoundError:
            print(f'[ERRO] Arquivo não encontrado: {self.caminho}')
            sys.exit(1)

        # Normalizar nomes de colunas (strip + lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        # Verificar colunas obrigatórias
        ausentes = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
        if ausentes:
            raise ValueError(
                f'Colunas obrigatórias ausentes na planilha: {ausentes}\n'
                f'Colunas encontradas: {list(df.columns)}'
            )

        # Colunas opcionais presentes no arquivo
        opcionais_presentes = [c for c in COLUNAS_OPCIONAIS if c in df.columns]

        empresas = []
        for _, row in df.iterrows():
            try:
                empresa = Empresa(
                    name=str(row['name']).strip(),
                    industry=str(row.get('industry', 'Unknown')).strip(),
                    scores=ScoreESG(
                        environment_score=int(row['environment_score']),
                        social_score=int(row['social_score']),
                        governance_score=int(row['governance_score']),
                    ),
                    # Campos opcionais — apenas preenchidos se presentes na planilha
                    cnpj=_str_ou_none(row.get('cnpj'))                         if 'cnpj' in opcionais_presentes else None,
                    email=_str_ou_none(row.get('email'))                       if 'email' in opcionais_presentes else None,
                    telefone=_str_ou_none(row.get('telefone'))                 if 'telefone' in opcionais_presentes else None,
                    contato=_str_ou_none(row.get('contato'))                   if 'contato' in opcionais_presentes else None,
                    quantidade_funcionarios=_int_ou_none(row.get('quantidade_funcionarios'))
                                                                               if 'quantidade_funcionarios' in opcionais_presentes else None,
                    endereco=_str_ou_none(row.get('endereco'))                 if 'endereco' in opcionais_presentes else None,
                    website=_str_ou_none(row.get('website'))                   if 'website' in opcionais_presentes else None,
                    descricao=_str_ou_none(row.get('descricao'))               if 'descricao' in opcionais_presentes else None,
                )
                empresas.append(empresa)
            except (ValueError, TypeError) as exc:
                print(f'[AVISO] Linha ignorada (name={row.get("name", "?")}): {exc}')

        return empresas
