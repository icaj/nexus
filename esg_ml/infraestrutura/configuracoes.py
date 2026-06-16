# esg_ml/infraestrutura/configuracoes.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto: dois níveis acima de esg_ml/infraestrutura/configuracoes.py
# Garante caminho absoluto independente do CWD ao executar scripts fora do Docker.
_RAIZ_PROJETO = Path(__file__).resolve().parents[2]


class Configuracoes(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    nome_aplicacao:      str  = 'ESG Nexus Enterprise'
    diretorio_artefatos: Path = _RAIZ_PROJETO / 'artifacts'

    # Caminhos derivados — dentro de artifacts/ para garantir permissão de escrita
    @property
    def pasta_bronze(self) -> Path:
        return self.diretorio_artefatos / 'bronze'

    @property
    def pasta_saida(self) -> Path:
        return self.diretorio_artefatos / 'processado'

    @property
    def pasta_resultados(self) -> Path:
        return self.diretorio_artefatos / 'resultados'

    # MLflow
    mlflow_tracking_uri: str = 'http://localhost:5000'
    mlflow_experimento:  str = 'esg-nexus-fornecedores'

    # JWT / segurança (Enterprise)
    segredo_jwt:                str = 'troque-esta-chave-em-producao'
    algoritmo_jwt:              str = 'HS256'
    minutos_expiracao_token:    int = 480
