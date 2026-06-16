from datetime import datetime, timedelta, timezone
import os
from typing import Any

import bcrypt
from jose import JWTError, jwt

CHAVE_SECRETA = os.getenv("SEGREDO_JWT", "troque-esta-chave-em-producao")
ALGORITMO = os.getenv("ALGORITMO_JWT", "HS256")
MINUTOS_EXPIRACAO = int(os.getenv("MINUTOS_EXPIRACAO_TOKEN", "480"))


def gerar_hash_senha(senha: str) -> str:
    if not senha:
        raise ValueError("A senha não pode ser vazia")
    return bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verificar_senha(senha: str, senha_hash: str) -> bool:
    if not senha or not senha_hash:
        return False
    try:
        return bcrypt.checkpw(senha.encode("utf-8"), senha_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def criar_token_acesso(dados: dict[str, Any]) -> str:
    payload = dados.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=MINUTOS_EXPIRACAO)
    return jwt.encode(payload, CHAVE_SECRETA, algorithm=ALGORITMO)


def decodificar_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, CHAVE_SECRETA, algorithms=[ALGORITMO])
    except JWTError:
        return None
