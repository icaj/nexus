# esg_ml/adaptadores/saida/repositorio_modelo_joblib.py
# Porta: RepositorioModelos → disco (.joblib via joblib)
# Salva três artefatos: modelo_knn, modelo_rf, config
import os, joblib
from pathlib import Path
from typing import Any

class RepositorioModeloJoblib:
    def __init__(self, pasta):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)

    def _path(self, nome: str) -> Path:
        return self.pasta / f'{nome}.joblib'

    def salvar(self, nome: str, modelo: Any, metadados: dict) -> None:
        joblib.dump({'modelo': modelo, **metadados}, self._path(nome))
        print(f'  ✓ {self._path(nome)}')

    def carregar(self, nome: str) -> tuple:
        payload = joblib.load(self._path(nome))
        modelo  = payload.pop('modelo', None)
        return modelo, payload

    def existe(self, nome: str) -> bool:
        return self._path(nome).exists()

    # Compatibilidade com interface Enterprise (carrega payload único)
    def carregar_payload(self, nome: str) -> dict:
        return joblib.load(self._path(nome))

    def salvar_payload(self, nome: str, payload: dict) -> None:
        joblib.dump(payload, self._path(nome))
