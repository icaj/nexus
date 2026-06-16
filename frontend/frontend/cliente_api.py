from __future__ import annotations

import os
from typing import Any

import requests


URL_API_PADRAO = os.getenv("ESG_NEXUS_API_URL", "http://localhost:8000")


class ErroAPI(RuntimeError):
    pass


class ClienteApiESG:
    def __init__(self, url_base: str = URL_API_PADRAO, token: str | None = None, timeout: int = 20) -> None:
        self.url_base = url_base.rstrip("/")
        self.token = token
        self.timeout = timeout

    @property
    def cabecalhos(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _tratar(self, resposta: requests.Response) -> Any:
        if resposta.status_code >= 400:
            try:
                detalhe = resposta.json().get("detail", resposta.text)
            except Exception:
                detalhe = resposta.text
            raise ErroAPI(str(detalhe))
        if not resposta.content:
            return None
        return resposta.json()

    def registrar(self, nome: str, email: str, senha: str, perfil: str) -> dict[str, Any]:
        resposta = requests.post(
            f"{self.url_base}/auth/registrar",
            json={"nome": nome, "email": email, "senha": senha, "perfil": perfil},
            timeout=self.timeout,
        )
        return self._tratar(resposta)

    def login(self, email: str, senha: str) -> dict[str, Any]:
        resposta = requests.post(
            f"{self.url_base}/auth/login",
            json={"email": email, "senha": senha},
            timeout=self.timeout,
        )
        return self._tratar(resposta)

    def meus_dados(self) -> dict[str, Any]:
        resposta = requests.get(f"{self.url_base}/auth/me", headers=self.cabecalhos, timeout=self.timeout)
        return self._tratar(resposta)

    def listar_fornecedores(self) -> list[dict[str, Any]]:
        resposta = requests.get(f"{self.url_base}/fornecedores", headers=self.cabecalhos, timeout=self.timeout)
        return self._tratar(resposta)

    def cadastrar_fornecedor(self, fornecedor: dict[str, Any]) -> dict[str, Any]:
        resposta = requests.post(
            f"{self.url_base}/fornecedores", json=fornecedor, headers=self.cabecalhos, timeout=self.timeout
        )
        return self._tratar(resposta)

    def classificar(self, fornecedor: dict[str, Any]) -> dict[str, Any]:
        resposta = requests.post(
            f"{self.url_base}/classificar", json=fornecedor, headers=self.cabecalhos, timeout=self.timeout
        )
        return self._tratar(resposta)


    def classificar_lote(self, fornecedores: list[dict[str, Any]]) -> dict[str, Any]:
        resposta = requests.post(
            f"{self.url_base}/classificar/lote",
            json={"fornecedores": fornecedores},
            headers=self.cabecalhos,
            timeout=max(self.timeout, 60),
        )
        return self._tratar(resposta)

    def explicar(self, fornecedor: dict[str, Any]) -> dict[str, Any]:
        resposta = requests.post(
            f"{self.url_base}/explicabilidade", json=fornecedor, headers=self.cabecalhos, timeout=self.timeout
        )
        return self._tratar(resposta)

    def dashboard_executivo(self) -> dict[str, Any]:
        resposta = requests.get(f"{self.url_base}/dashboard/executivo", headers=self.cabecalhos, timeout=self.timeout)
        return self._tratar(resposta)

    def dashboard_ml(self) -> dict[str, Any]:
        resposta = requests.get(f"{self.url_base}/dashboard/ml", headers=self.cabecalhos, timeout=self.timeout)
        return self._tratar(resposta)

    def dashboard_estatistico(
        self,
        data_inicio: str | None = None,
        data_fim: str | None = None,
        setor: str | None = None,
        maturidade: str | None = None,
    ) -> dict[str, Any]:
        params = {k: v for k, v in {
            'data_inicio': data_inicio, 'data_fim': data_fim,
            'setor': setor, 'maturidade': maturidade,
        }.items() if v is not None}
        resposta = requests.get(
            f"{self.url_base}/dashboard/estatistico",
            params=params,
            headers=self.cabecalhos,
            timeout=max(self.timeout, 30),
        )
        return self._tratar(resposta)
