import logging
from typing import Any


class RegistradorPadrao:
    def __init__(self, nome: str) -> None:
        self.registrador = logging.getLogger(nome)

    def info(self, evento: str, **dados: Any) -> None:
        self.registrador.info("%s %s", evento, dados)

    def warning(self, evento: str, **dados: Any) -> None:
        self.registrador.warning("%s %s", evento, dados)


def configurar_registro_log() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    try:
        import structlog

        structlog.configure(
            processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )
    except ModuleNotFoundError:
        pass


def obter_registrador(nome: str):
    try:
        import structlog

        return structlog.get_logger(nome)
    except ModuleNotFoundError:
        return RegistradorPadrao(nome)
