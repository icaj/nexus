"""
Testes funcionais do ESG Nexus Enterprise — executa contra a API em URL_API.

Uso:
    python testar_funcionalidades.py                      # API local padrão
    python testar_funcionalidades.py http://IP:8000       # servidor remoto
"""
from __future__ import annotations

import io
import json
import sys
import time
from typing import Any

import requests

URL_API = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://35.212.103.210:8000"

# ── Cores para output ─────────────────────────────────────────────────────────
OK    = "\033[32m✓\033[0m"
FAIL  = "\033[31m✗\033[0m"
INFO  = "\033[36m·\033[0m"
WARN  = "\033[33m!\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

_resultados: list[dict] = []


def _log(simbolo: str, titulo: str, detalhe: str = "") -> None:
    msg = f"  {simbolo} {titulo}"
    if detalhe:
        msg += f"\n      {detalhe}"
    print(msg)


def _ok(titulo: str, detalhe: str = "") -> None:
    _log(OK, titulo, detalhe)
    _resultados.append({"status": "ok", "titulo": titulo})


def _fail(titulo: str, detalhe: str = "") -> None:
    _log(FAIL, titulo, detalhe)
    _resultados.append({"status": "fail", "titulo": titulo, "detalhe": detalhe})


def _info(msg: str) -> None:
    print(f"  {INFO} {msg}")


def secao(titulo: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {titulo}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ── Payload de fornecedor de teste ────────────────────────────────────────────
FORN_BAIXO_RISCO = {
    "codigo_fornecedor": "TEST-001",
    "razao_social": "Empresa Verde LTDA",
    "cnpj": "11.111.111/0001-11",
    "setor": "tecnologia",
    "pais": "BR",
    "possui_politica_ambiental": True,
    "emissoes_carbono_ton": 50.0,
    "percentual_energia_renovavel": 80.0,
    "percentual_reciclagem_residuos": 75.0,
    "incidentes_trabalhistas_12m": 0,
    "possui_programa_diversidade": True,
    "possui_politica_privacidade_dados": True,
    "possui_politica_anticorrupcao": True,
    "consta_lista_sancoes": False,
    "noticias_negativas_12m": 0,
    "quantidade_certificacoes": 3,
    "receita_anual": 5000000.0,
}

FORN_ALTO_RISCO = {
    "codigo_fornecedor": "TEST-002",
    "razao_social": "Empresa Cinza LTDA",
    "cnpj": "22.222.222/0002-22",
    "setor": "industria",
    "pais": "BR",
    "possui_politica_ambiental": False,
    "emissoes_carbono_ton": 4500.0,
    "percentual_energia_renovavel": 5.0,
    "percentual_reciclagem_residuos": 10.0,
    "incidentes_trabalhistas_12m": 8,
    "possui_programa_diversidade": False,
    "possui_politica_privacidade_dados": False,
    "possui_politica_anticorrupcao": False,
    "consta_lista_sancoes": True,
    "noticias_negativas_12m": 6,
    "quantidade_certificacoes": 0,
    "receita_anual": 800000.0,
}

TOKEN: str | None = None

ADMIN_EMAIL = "admin@esgnexus.example"
ADMIN_SENHA = "admin123"


# ─────────────────────────────────────────────────────────────────────────────
# F0 — Pré-voo: treinar modelos se ausentes
# ─────────────────────────────────────────────────────────────────────────────
def _modelos_carregados() -> bool:
    try:
        r = requests.get(f"{URL_API}/saude", timeout=10)
        if r.status_code != 200:
            return False
        d = r.json()
        return all([d.get("modelo_knn_carregado"), d.get("modelo_rf_carregado"),
                    d.get("config_carregado")])
    except Exception:
        return False


def preflight_treinar_se_necessario() -> None:
    if _modelos_carregados():
        return

    print(f"\n{BOLD}  [pré-voo] Modelos ausentes — disparando treinamento inicial...{RESET}")

    # Login como admin semeado no startup
    try:
        r_login = requests.post(f"{URL_API}/auth/login",
                                json={"email": ADMIN_EMAIL, "senha": ADMIN_SENHA},
                                timeout=10)
    except requests.ConnectionError:
        print(f"  {FAIL} Não foi possível conectar em {URL_API}")
        sys.exit(1)

    if r_login.status_code != 200:
        print(f"  {WARN} Login admin falhou ({r_login.status_code}) — treinamento pulado.")
        print(f"       Verifique USUARIO_ADMIN_EMAIL / USUARIO_ADMIN_SENHA no .env do servidor.")
        return

    token_admin = r_login.json()["access_token"]
    r_treinar = requests.post(f"{URL_API}/treinar",
                              headers={"Authorization": f"Bearer {token_admin}"},
                              timeout=600)
    if r_treinar.status_code == 200:
        print(f"  {OK} Treinamento concluído: {r_treinar.json().get('melhor_modelo','?')}")
    else:
        print(f"  {FAIL} POST /treinar retornou {r_treinar.status_code}: {r_treinar.text[:200]}")

    if not _modelos_carregados():
        print(f"  {WARN} Modelos ainda não visíveis em /saude — aguardando 5s...")
        time.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Saúde da API
# ─────────────────────────────────────────────────────────────────────────────
def testar_saude() -> None:
    secao("F1 · Saúde da API")
    try:
        r = requests.get(f"{URL_API}/saude", timeout=10)
        if r.status_code == 200:
            d = r.json()
            _ok("GET /saude retornou 200")
            for k, v in d.items():
                simbolo = OK if v else FAIL
                _log(simbolo, f"  {k}: {v}")
                if not v:
                    _fail(f"Artefato ausente: {k}")
        else:
            _fail(f"GET /saude retornou {r.status_code}", r.text[:200])
    except requests.ConnectionError:
        _fail(f"Não foi possível conectar em {URL_API}", "Verifique se a API está em execução.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Autenticação
# ─────────────────────────────────────────────────────────────────────────────
def testar_auth() -> str | None:
    global TOKEN
    secao("F2 · Autenticação")
    email = f"teste.funcional.{int(time.time())}@example.com"
    senha = "Teste@12345"

    # Registro
    r = requests.post(f"{URL_API}/auth/registrar", json={
        "nome": "Usuário Teste Funcional",
        "email": email,
        "senha": senha,
        "perfil": "administrador",
    }, timeout=10)
    if r.status_code == 201:
        _ok("POST /auth/registrar — usuário criado")
    elif r.status_code == 409:
        _info("Usuário já existe (409) — prosseguindo com login")
    else:
        _fail(f"POST /auth/registrar retornou {r.status_code}", r.text[:200])
        return None

    # Login
    r = requests.post(f"{URL_API}/auth/login", json={"email": email, "senha": senha}, timeout=10)
    if r.status_code == 200:
        TOKEN = r.json()["access_token"]
        perfil = r.json()["usuario"]["perfil"]
        _ok(f"POST /auth/login — token obtido (perfil: {perfil})")
    else:
        _fail(f"POST /auth/login retornou {r.status_code}", r.text[:200])
        return None

    # Me
    r = requests.get(f"{URL_API}/auth/me", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
    if r.status_code == 200:
        _ok(f"GET /auth/me — {r.json()['email']}")
    else:
        _fail(f"GET /auth/me retornou {r.status_code}", r.text[:200])

    return TOKEN


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Classificação individual
# ─────────────────────────────────────────────────────────────────────────────
def testar_classificacao_individual() -> None:
    secao("F3 · Classificação individual (POST /classificar)")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}

    for label, payload in [("baixo risco", FORN_BAIXO_RISCO), ("alto risco", FORN_ALTO_RISCO)]:
        r = requests.post(f"{URL_API}/classificar", json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            d = r.json()
            nivel   = d.get("nivel_risco", "?")
            pontuacao = d.get("pontuacao_esg", "?")
            _ok(f"Classificou '{payload['razao_social']}' — nível: {nivel}, score ESG: {pontuacao}")
            # Verifica campos obrigatórios na resposta
            for campo in ["pontuacao_ambiental", "pontuacao_social", "pontuacao_governanca",
                          "nivel_risco", "recomendacao", "probabilidade_ml_alto_risco"]:
                if campo not in d or d[campo] is None:
                    _fail(f"Campo ausente na resposta: {campo}")
        else:
            _fail(f"POST /classificar ({label}) retornou {r.status_code}", r.text[:300])


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Persistência (fornecedores no banco)
# ─────────────────────────────────────────────────────────────────────────────
def testar_persistencia() -> None:
    secao("F4 · Persistência no banco (GET /fornecedores)")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(f"{URL_API}/fornecedores", headers=headers, timeout=10)
    if r.status_code == 200:
        lista = r.json()
        _ok(f"GET /fornecedores retornou {len(lista)} fornecedor(es)")
        # Verifica se os fornecedores de teste foram gravados
        nomes = [f.get("name", "") for f in lista]
        for esperado in ["Empresa Verde LTDA", "Empresa Cinza LTDA"]:
            if any(esperado in n for n in nomes):
                _ok(f"  Fornecedor '{esperado}' encontrado no banco")
            else:
                _fail(f"  Fornecedor '{esperado}' NÃO encontrado no banco (gravação falhou)")
    else:
        _fail(f"GET /fornecedores retornou {r.status_code}", r.text[:200])


# ─────────────────────────────────────────────────────────────────────────────
# F5 — Classificação em lote (JSON)
# ─────────────────────────────────────────────────────────────────────────────
def testar_lote() -> None:
    secao("F5 · Classificação em lote (POST /classificar/lote)")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}
    lote = []
    for i in range(3):
        f = dict(FORN_BAIXO_RISCO)
        f["codigo_fornecedor"] = f"LOTE-{i+1:03d}"
        f["razao_social"] = f"Fornecedor Lote {i+1}"
        f["cnpj"] = f"33.{i:03d}.111/0001-{i:02d}"
        lote.append(f)

    r = requests.post(f"{URL_API}/classificar/lote",
                      json={"fornecedores": lote},
                      headers=headers, timeout=60)
    if r.status_code == 200:
        d = r.json()
        processados = d.get("total_processados", 0)
        erros_count = d.get("total_erros", 0)
        _ok(f"POST /classificar/lote — processados: {processados}, erros: {erros_count}")
        if processados < 3:
            _fail(f"Esperado ≥3 processados, obtido {processados}")
    elif r.status_code == 503:
        _fail(f"POST /classificar/lote — modelos não treinados (503)", r.json().get("detail", "")[:120])
    else:
        _fail(f"POST /classificar/lote retornou {r.status_code}", r.text[:300])


# ─────────────────────────────────────────────────────────────────────────────
# F6 — Upload de planilha
# ─────────────────────────────────────────────────────────────────────────────
def testar_upload() -> None:
    secao("F6 · Upload de planilha (POST /avaliar/upload)")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Gera CSV em memória
    linhas = [
        "codigo_fornecedor,razao_social,cnpj,setor,pais,"
        "possui_politica_ambiental,emissoes_carbono_ton,percentual_energia_renovavel,"
        "percentual_reciclagem_residuos,incidentes_trabalhistas_12m,"
        "possui_programa_diversidade,possui_politica_privacidade_dados,"
        "possui_politica_anticorrupcao,consta_lista_sancoes,"
        "noticias_negativas_12m,quantidade_certificacoes,receita_anual",
        "UP-001,Upload Teste Alpha,44.444.444/0001-44,servicos,BR,"
        "sim,80,70,65,0,sim,sim,sim,nao,0,2,2000000",
        "UP-002,Upload Teste Beta,55.555.555/0002-55,energia,BR,"
        "nao,2000,20,15,2,nao,nao,nao,nao,1,0,500000",
    ]
    csv_bytes = "\n".join(linhas).encode("utf-8")

    r = requests.post(
        f"{URL_API}/avaliar/upload",
        files={"arquivo": ("teste.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers=headers,
        timeout=60,
    )
    if r.status_code == 200:
        resultados = r.json()
        _ok(f"POST /avaliar/upload — {len(resultados)} resultado(s) retornado(s)")
        for res in resultados:
            _info(f"  {res.get('razao_social')} → {res.get('nivel_risco')} / {res.get('pontuacao_esg')}")
    else:
        _fail(f"POST /avaliar/upload retornou {r.status_code}", r.text[:300])


# ─────────────────────────────────────────────────────────────────────────────
# F7 — Dashboard Executivo
# ─────────────────────────────────────────────────────────────────────────────
def testar_dashboard_executivo() -> None:
    secao("F7 · Dashboard Executivo (GET /dashboard/executivo)")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(f"{URL_API}/dashboard/executivo", headers=headers, timeout=15)
    if r.status_code == 200:
        d = r.json()
        kpis = d.get("kpis", {})
        _ok("GET /dashboard/executivo retornou 200")
        _info(f"  total_fornecedores: {kpis.get('total_fornecedores')}")
        _info(f"  score_medio: {kpis.get('score_medio')}")
        _info(f"  alto_risco: {kpis.get('alto_risco')}")
        _info(f"  distribuicao_risco: {d.get('distribuicao_risco')}")
        for campo in ["kpis", "distribuicao_risco", "medias_pilares", "top_risco", "melhores"]:
            if campo not in d:
                _fail(f"Campo ausente: {campo}")
    else:
        _fail(f"GET /dashboard/executivo retornou {r.status_code}", r.text[:200])


# ─────────────────────────────────────────────────────────────────────────────
# F8 — Dashboard ML
# ─────────────────────────────────────────────────────────────────────────────
def testar_dashboard_ml() -> None:
    secao("F8 · Dashboard ML (GET /dashboard/ml)")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(f"{URL_API}/dashboard/ml", headers=headers, timeout=15)
    if r.status_code == 200:
        d = r.json()
        _ok("GET /dashboard/ml retornou 200")
        metricas = d.get("metricas", {})
        _info(f"  accuracy: {metricas.get('accuracy')}  f1: {metricas.get('f1')}")
        _info(f"  feature_importance: {len(d.get('feature_importance', []))} variáveis")
        _info(f"  dispersao: {len(d.get('dispersao', []))} pontos")
    else:
        _fail(f"GET /dashboard/ml retornou {r.status_code}", r.text[:200])


# ─────────────────────────────────────────────────────────────────────────────
# F9 — Histórico de avaliações
# ─────────────────────────────────────────────────────────────────────────────
def testar_historico() -> None:
    secao("F9 · Histórico de avaliações")
    if not TOKEN:
        _fail("Sem token — pulando")
        return

    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Pega o primeiro fornecedor disponível
    r = requests.get(f"{URL_API}/fornecedores", headers=headers, timeout=10)
    if r.status_code != 200 or not r.json():
        _fail("Nenhum fornecedor disponível para testar histórico")
        return

    forn_id = r.json()[0]["id"]
    r2 = requests.get(f"{URL_API}/fornecedores/{forn_id}/historico", headers=headers, timeout=10)
    if r2.status_code == 200:
        h = r2.json()
        _ok(f"GET /fornecedores/{forn_id}/historico — {h.get('total_avaliacoes')} avaliação(ões)")
    else:
        _fail(f"GET /fornecedores/{forn_id}/historico retornou {r2.status_code}", r2.text[:200])

    # Plano de ação
    r3 = requests.get(f"{URL_API}/fornecedores/{forn_id}/plano-acao", headers=headers, timeout=10)
    if r3.status_code == 200:
        _ok(f"GET /fornecedores/{forn_id}/plano-acao — {len(r3.json())} ação(ões)")
    else:
        _fail(f"GET /fornecedores/{forn_id}/plano-acao retornou {r3.status_code}", r3.text[:200])


# ─────────────────────────────────────────────────────────────────────────────
# Resumo
# ─────────────────────────────────────────────────────────────────────────────
def resumo() -> None:
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  RESUMO{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    oks   = [r for r in _resultados if r["status"] == "ok"]
    fails = [r for r in _resultados if r["status"] == "fail"]
    print(f"  {OK} Passou : {len(oks)}")
    print(f"  {FAIL} Falhou: {len(fails)}")
    if fails:
        print(f"\n  Falhas:")
        for f in fails:
            detalhe = f" — {f['detalhe'][:80]}" if f.get("detalhe") else ""
            print(f"    {FAIL} {f['titulo']}{detalhe}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}ESG Nexus — Testes Funcionais{RESET}")
    print(f"  API: {BOLD}{URL_API}{RESET}\n")

    preflight_treinar_se_necessario()
    testar_saude()
    testar_auth()
    testar_classificacao_individual()
    testar_persistencia()
    testar_lote()
    testar_upload()
    testar_dashboard_executivo()
    testar_dashboard_ml()
    testar_historico()
    resumo()
