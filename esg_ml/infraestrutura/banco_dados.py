from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# Fallback para NeonDB; sobreposto por DATABASE_URL no .env quando presente
URL_BANCO_PADRAO = "postgresql+psycopg://neondb_owner:npg_oQ4iTnNEqI3x@ep-fancy-art-apt303j2-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


def _raiz_projeto() -> Path:
    return Path(__file__).resolve().parents[2]


def carregar_variaveis_ambiente() -> None:
    caminho_env = _raiz_projeto() / ".env"
    if caminho_env.exists():
        load_dotenv(dotenv_path=caminho_env)  # removido override=True


carregar_variaveis_ambiente()


class Base(DeclarativeBase):
    __allow_unmapped__ = True 


def _obter_url_banco() -> str:
    return os.getenv("DATABASE_URL", URL_BANCO_PADRAO)


def obter_url_banco_mascarada() -> str:
    return make_url(_obter_url_banco()).render_as_string(hide_password=True)


_PSYCOPG3_SSL_PARAMS = {"sslmode", "sslrootcert", "sslcert", "sslkey", "channel_binding"}


def _argumentos_engine(url_banco: str) -> dict:
    url = make_url(url_banco)
    argumentos: dict = {"pool_pre_ping": True}
    if url.drivername.startswith("sqlite"):
        argumentos["connect_args"] = {"check_same_thread": False}
        return argumentos
    argumentos["pool_size"] = int(os.getenv("DATABASE_POOL_SIZE", "5"))
    argumentos["max_overflow"] = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
    # Repassa para o driver psycopg3 todos os parâmetros SSL presentes na URL
    # (sslmode, channel_binding, etc.) via connect_args, evitando duplicação.
    ssl_params = {k: v for k, v in url.query.items() if k in _PSYCOPG3_SSL_PARAMS}
    if ssl_params:
        argumentos["connect_args"] = ssl_params
    return argumentos


URL_BANCO = _obter_url_banco()
engine = create_engine(URL_BANCO, **_argumentos_engine(URL_BANCO))
SessaoLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def obter_sessao() -> Generator[Session, None, None]:
    sessao = SessaoLocal()
    try:
        yield sessao
    except Exception:
        sessao.rollback()
        raise
    finally:
        sessao.close()


def criar_tabelas() -> None:
    from esg_ml.infraestrutura import modelos_banco  # noqa: F401

    Base.metadata.create_all(bind=engine)
