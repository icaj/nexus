from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from esg_ml.infraestrutura.banco_dados import obter_sessao
from esg_ml.infraestrutura.modelos_banco import UsuarioBanco
from esg_ml.infraestrutura.seguranca import decodificar_token

bearer = HTTPBearer(auto_error=False)

def obter_usuario_atual(
    credenciais: HTTPAuthorizationCredentials | None = Depends(bearer),
    sessao: Session = Depends(obter_sessao),
) -> UsuarioBanco:
    if credenciais is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token ausente')
    payload = decodificar_token(credenciais.credentials)
    if not payload or 'sub' not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token inválido')
    usuario = sessao.query(UsuarioBanco).filter(UsuarioBanco.email == payload['sub']).first()
    if usuario is None or not usuario.ativo:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Usuário inativo ou inexistente')
    return usuario

def exigir_perfil(*perfis: str):
    def _dep(usuario: UsuarioBanco = Depends(obter_usuario_atual)) -> UsuarioBanco:
        if usuario.perfil not in perfis:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Perfil sem autorização')
        return usuario
    return _dep
