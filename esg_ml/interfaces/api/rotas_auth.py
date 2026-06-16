from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from esg_ml.infraestrutura.banco_dados import obter_sessao
from esg_ml.infraestrutura.modelos_banco import UsuarioBanco
from esg_ml.infraestrutura.seguranca import criar_token_acesso, gerar_hash_senha, verificar_senha
from esg_ml.interfaces.api.dependencias_auth import obter_usuario_atual
from esg_ml.interfaces.api.esquemas_auth import LoginEntrada, TokenSaida, UsuarioCriacao, UsuarioSaida

roteador = APIRouter(prefix='/auth', tags=['Autenticação'])

def _saida(usuario: UsuarioBanco) -> UsuarioSaida:
    return UsuarioSaida(id=usuario.id, nome=usuario.nome, email=usuario.email, perfil=usuario.perfil, ativo=usuario.ativo)

@roteador.post('/registrar', response_model=UsuarioSaida, status_code=201)
def registrar_usuario(entrada: UsuarioCriacao, sessao: Session = Depends(obter_sessao)) -> UsuarioSaida:
    existente = sessao.query(UsuarioBanco).filter(UsuarioBanco.email == entrada.email).first()
    if existente:
        raise HTTPException(status_code=409, detail='E-mail já cadastrado')
    usuario = UsuarioBanco(nome=entrada.nome, email=str(entrada.email), senha_hash=gerar_hash_senha(entrada.senha), perfil=entrada.perfil)
    sessao.add(usuario); sessao.commit(); sessao.refresh(usuario)
    return _saida(usuario)

@roteador.post('/login', response_model=TokenSaida)
def login(entrada: LoginEntrada, sessao: Session = Depends(obter_sessao)) -> TokenSaida:
    usuario = sessao.query(UsuarioBanco).filter(UsuarioBanco.email == entrada.email).first()
    if usuario is None or not verificar_senha(entrada.senha, usuario.senha_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Credenciais inválidas')
    token = criar_token_acesso({'sub': usuario.email, 'perfil': usuario.perfil})
    return TokenSaida(access_token=token, usuario=_saida(usuario))

@roteador.get('/me', response_model=UsuarioSaida)
def meus_dados(usuario: UsuarioBanco = Depends(obter_usuario_atual)) -> UsuarioSaida:
    return _saida(usuario)
