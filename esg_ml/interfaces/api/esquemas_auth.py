from pydantic import BaseModel, EmailStr, Field

class UsuarioCriacao(BaseModel):
    nome: str = Field(min_length=3, max_length=120)
    email: EmailStr
    senha: str = Field(min_length=6, max_length=80)
    perfil: str = 'analista_esg'

class UsuarioSaida(BaseModel):
    id: int
    nome: str
    email: EmailStr
    perfil: str
    ativo: bool

class LoginEntrada(BaseModel):
    email: EmailStr
    senha: str

class TokenSaida(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    usuario: UsuarioSaida
