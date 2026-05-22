import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv


ARQUIVO_BASE_PROCESSADA = Path("./data/bronze/base_processada.csv")
ARQUIVO_PESOS_INDUSTRIA = Path("./data/bronze/pesos_por_industria.csv")
#ARQUIVO_BRUTO = Path("./data/raw/data.csv")

SCHEMA_NAME = "public"
TABELA_BASE_PROCESSADA = "esg_base_processada"
TABELA_PESOS_INDUSTRIA = "esg_pesos_industria"
#TABELA_DADOS_BRUTOS = "esg_dados_brutos"


def obter_database_url() -> str:
    """
    Carrega a variável DATABASE_URL do ambiente ou de um arquivo .env.
    """
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("[ERRO] Variável DATABASE_URL não encontrada.")
        print("Crie um arquivo .env ou configure a variável de ambiente DATABASE_URL.")
        sys.exit(1)

    return database_url


def validar_arquivo(caminho: Path) -> None:
    """
    Verifica se o arquivo CSV existe.
    """
    if not caminho.exists():
        print(f"[ERRO] Arquivo não encontrado: {caminho}")
        return -1
    else:
        return 0


def carregar_base_processada(caminho: Path) -> pd.DataFrame:
    """
    Carrega e valida o CSV principal de fornecedores/scores ESG.
    """
    df = pd.read_csv(caminho, sep=",", encoding="utf-8")

    colunas_obrigatorias = [
        "name",
        "industry",
        "environment_score",
        "social_score",
        "governance_score",
        "total_score",
        "score_ponderado",
        "total_grade",
        "total_level",
        "maturidade",
        "risco",
        "impacto",
        "quadrante",
    ]

    colunas_ausentes = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes em {caminho}: {', '.join(colunas_ausentes)}"
        )

    df = df[colunas_obrigatorias].copy()

    colunas_int = [
        "environment_score",
        "social_score",
        "governance_score",
        "total_score",
    ]

    colunas_float = [
        "score_ponderado",
        "risco",
        "impacto",
    ]

    for col in colunas_int:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    for col in colunas_float:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")

    colunas_texto = [
        "name",
        "industry",
        "total_grade",
        "total_level",
        "maturidade",
        "quadrante",
    ]

    for col in colunas_texto:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def carregar_pesos_industria(caminho: Path) -> pd.DataFrame:
    """
    Carrega e valida o CSV de pesos por indústria.

    Observação:
    O arquivo recebido possui a indústria na coluna 'Unnamed: 0'.
    O script renomeia essa coluna para 'industry'.
    """
    df = pd.read_csv(caminho, sep=",", encoding="utf-8")

    if "industry" not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "industry"})

    colunas_obrigatorias = [
        "industry",
        "w_E",
        "w_S",
        "w_G",
        "n",
        "fonte",
    ]

    colunas_ausentes = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes em {caminho}: {', '.join(colunas_ausentes)}"
        )

    df = df[colunas_obrigatorias].copy()

    df["industry"] = df["industry"].fillna("").astype(str).str.strip()
    df["fonte"] = df["fonte"].fillna("").astype(str).str.strip()

    df["w_E"] = pd.to_numeric(df["w_E"], errors="raise").astype("float64")
    df["w_S"] = pd.to_numeric(df["w_S"], errors="raise").astype("float64")
    df["w_G"] = pd.to_numeric(df["w_G"], errors="raise").astype("float64")
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype("int64")

    return df


#def carregar_dados_brutos(caminho: Path) -> pd.DataFrame:
#    df = pd.read_csv(caminho, sep=",", encoding="utf-8")

#    colunas_int = [
#        "environment_score",
#        "social_score",
#        "governance_score",
#        "total_score",
#    ]

#    colunas_float = [
#        "score_ponderado",
#        "risco",
#        "impacto",
#    ]

#    for col in colunas_int:
#        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

#    for col in colunas_float:
#        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")

#    colunas_texto = [
#        "name",
#        "industry",
#        "total_grade",
#        "total_level",
#        "maturidade",
#        "quadrante",
#    ]

#    for col in colunas_texto:
#       df[col] = df[col].fillna("").astype(str).str.strip()

#    return df


def criar_tabelas_e_indices(conn: psycopg.Connection) -> None:
    """
    Cria as tabelas e os índices necessários no PostgreSQL/NeonDB.
    """
    sql = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (
        id BIGSERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        industry TEXT NOT NULL,
        environment_score INTEGER NOT NULL,
        social_score INTEGER NOT NULL,
        governance_score INTEGER NOT NULL,
        total_score INTEGER NOT NULL,
        score_ponderado NUMERIC(10, 2) NOT NULL,
        total_grade VARCHAR(20),
        total_level VARCHAR(50),
        maturidade VARCHAR(50),
        risco NUMERIC(10, 2),
        impacto NUMERIC(10, 2),
        quadrante VARCHAR(100),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        CONSTRAINT chk_esg_environment_score
            CHECK (environment_score BETWEEN 0 AND 1000),

        CONSTRAINT chk_esg_social_score
            CHECK (social_score BETWEEN 0 AND 1000),

        CONSTRAINT chk_esg_governance_score
            CHECK (governance_score BETWEEN 0 AND 1000),

        CONSTRAINT chk_esg_total_score
            CHECK (total_score BETWEEN 0 AND 3000)
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{TABELA_PESOS_INDUSTRIA} (
        industry TEXT PRIMARY KEY,
        w_e NUMERIC(12, 6) NOT NULL,
        w_s NUMERIC(12, 6) NOT NULL,
        w_g NUMERIC(12, 6) NOT NULL,
        n INTEGER NOT NULL,
        fonte TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        CONSTRAINT chk_pesos_w_e CHECK (w_e >= 0 AND w_e <= 1),
        CONSTRAINT chk_pesos_w_s CHECK (w_s >= 0 AND w_s <= 1),
        CONSTRAINT chk_pesos_w_g CHECK (w_g >= 0 AND w_g <= 1),
        CONSTRAINT chk_pesos_n CHECK (n >= 0)
    );

    CREATE INDEX IF NOT EXISTS idx_esg_base_industry
        ON {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (industry);

    CREATE INDEX IF NOT EXISTS idx_esg_base_name
        ON {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (name);

    CREATE INDEX IF NOT EXISTS idx_esg_base_total_score
        ON {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (total_score DESC);

    CREATE INDEX IF NOT EXISTS idx_esg_base_score_ponderado
        ON {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (score_ponderado DESC);

    CREATE INDEX IF NOT EXISTS idx_esg_base_maturidade
        ON {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (maturidade);

    CREATE INDEX IF NOT EXISTS idx_esg_base_quadrante
        ON {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} (quadrante);
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()


def limpar_tabelas(conn: psycopg.Connection) -> None:
    """
    Limpa as tabelas antes da importação.
    """
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {SCHEMA_NAME}.{TABELA_BASE_PROCESSADA} RESTART IDENTITY;")
        cur.execute(f"TRUNCATE TABLE {SCHEMA_NAME}.{TABELA_PESOS_INDUSTRIA};")

    conn.commit()


def copy_dataframe(conn: psycopg.Connection, df: pd.DataFrame, tabela: str, colunas: list[str]) -> None:
    """
    Importa um DataFrame para o PostgreSQL usando COPY.
    """
    buffer = StringIO()
    df[colunas].to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    colunas_sql = ", ".join(colunas)

    copy_sql = f"""
        COPY {SCHEMA_NAME}.{tabela} ({colunas_sql})
        FROM STDIN
        WITH (FORMAT CSV)
    """

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            copy.write(buffer.getvalue())

    conn.commit()


def importar_base_processada(conn: psycopg.Connection, df: pd.DataFrame) -> None:
    """
    Importa o CSV base_processada.csv para a tabela esg_base_processada.
    """
    colunas = [
        "name",
        "industry",
        "environment_score",
        "social_score",
        "governance_score",
        "total_score",
        "score_ponderado",
        "total_grade",
        "total_level",
        "maturidade",
        "risco",
        "impacto",
        "quadrante",
    ]

    copy_dataframe(conn, df, TABELA_BASE_PROCESSADA, colunas)


def importar_pesos_industria(conn: psycopg.Connection, df: pd.DataFrame) -> None:
    """
    Importa o CSV pesos_por_industria.csv para a tabela esg_pesos_industria.
    """
    df = df.rename(
        columns={
            "w_E": "w_e",
            "w_S": "w_s",
            "w_G": "w_g",
        }
    )

    colunas = [
        "industry",
        "w_e",
        "w_s",
        "w_g",
        "n",
        "fonte",
    ]

    copy_dataframe(conn, df, TABELA_PESOS_INDUSTRIA, colunas)


def main() -> None:
    if validar_arquivo(ARQUIVO_BASE_PROCESSADA) == -1:
        print(f"Arquivo {ARQUIVO_BASE_PROCESSADA} não encontrado. Verifique o caminho e tente novamente.")
        return
    
    if validar_arquivo(ARQUIVO_PESOS_INDUSTRIA) == -1:
        print(f"Arquivo {ARQUIVO_PESOS_INDUSTRIA} não encontrado. Verifique o caminho e tente novamente.")
        return
    
#    if validar_arquivo(ARQUIVO_BRUTO) == -1:
#        print(f"Arquivo {ARQUIVO_BRUTO} não encontrado. Verifique o caminho e tente novamente.")
#        return

    print("Carregando CSVs...")
    df_base = carregar_base_processada(ARQUIVO_BASE_PROCESSADA)
    df_pesos = carregar_pesos_industria(ARQUIVO_PESOS_INDUSTRIA)
#    df_bruto = carregar_dados_brutos(ARQUIVO_BRUTO)

    print(f"Registros base_processada.csv: {len(df_base)}")
    print(f"Registros pesos_por_industria.csv: {len(df_pesos)}")

    database_url = obter_database_url()

    print("Conectando ao NeonDB/PostgreSQL...")

    try:
        with psycopg.connect(database_url) as conn:
            print("Criando tabelas e índices...")
            criar_tabelas_e_indices(conn)

            print("Limpando tabelas...")
            limpar_tabelas(conn)

            print("Importando pesos por indústria...")
            importar_pesos_industria(conn, df_pesos)

            print("Importando base processada...")
            importar_base_processada(conn, df_base)

            print("Importação concluída com sucesso.")

    except Exception as e:
        print(f"[ERRO] Falha na importação: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
