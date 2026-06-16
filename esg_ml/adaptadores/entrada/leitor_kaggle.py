# esg_ml/adaptadores/entrada/leitor_kaggle.py
import os
import pandas as pd


class LeitorKaggle:
    """Lê a base de dados ESG tratada em data/raw/data.csv."""

    def carregar(self, destino: str) -> pd.DataFrame:
        if not os.path.exists(destino):
            raise FileNotFoundError(
                f"Base de dados não encontrada: {destino}\n"
                "Verifique se o arquivo data/raw/data.csv está presente no repositório."
            )
        return pd.read_csv(destino)
