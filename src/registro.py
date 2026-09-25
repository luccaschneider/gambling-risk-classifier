"""
Registro auditável das classificações.

Grava em SQLite cada classificação atendida pela API: entradas, resultado,
intervenção e versão do modelo. Serve para auditar decisões — saber o que foi
decidido, quando, com quais dados e por qual versão.

O banco fica num diretório temporário porque o plano gratuito do Render não
tem disco persistente: o arquivo desaparece quando o serviço reinicia ou
hiberna. Em uso real, isto seria uma tabela no banco da operadora.

A tabela é podada para as ENTRADAS_MAXIMAS mais recentes, de modo que a
demonstração pública não cresça sem controle.
"""
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ENTRADAS_MAXIMAS = 500

_CAMINHO_BANCO = Path(os.environ.get(
    "REGISTRO_BANCO",
    Path(tempfile.gettempdir()) / "classificacoes_risco.sqlite3",
))

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS classificacoes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em          TEXT NOT NULL,
    usuario_externo    TEXT,
    entradas           TEXT NOT NULL,
    classe_prevista    TEXT NOT NULL,
    probabilidades     TEXT NOT NULL,
    indice_risco       REAL NOT NULL,
    intervencao_codigo TEXT NOT NULL,
    avisos             TEXT NOT NULL,
    versao_modelo      TEXT NOT NULL
);
"""


def _conectar():
    conexao = sqlite3.connect(_CAMINHO_BANCO, timeout=10)
    conexao.row_factory = sqlite3.Row
    return conexao


def iniciar():
    """Cria o banco e a tabela, se ainda não existirem."""
    with _conectar() as conexao:
        conexao.execute(_ESQUEMA)


def gravar(entradas, resultado, indice_risco, versao_modelo, usuario_externo=None):
    """Grava uma classificação e devolve o id gerado."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conectar() as conexao:
        conexao.execute(_ESQUEMA)
        cursor = conexao.execute(
            """INSERT INTO classificacoes
               (criado_em, usuario_externo, entradas, classe_prevista, probabilidades,
                indice_risco, intervencao_codigo, avisos, versao_modelo)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                agora,
                usuario_externo,
                json.dumps(entradas, ensure_ascii=False),
                resultado["classe_prevista"],
                json.dumps(resultado["probabilidades"], ensure_ascii=False),
                float(indice_risco),
                resultado["intervencao"]["codigo"],
                json.dumps(resultado["avisos"], ensure_ascii=False),
                versao_modelo,
            ),
        )
        novo_id = cursor.lastrowid
        # mantem apenas as ENTRADAS_MAXIMAS mais recentes
        conexao.execute(
            """DELETE FROM classificacoes
               WHERE id NOT IN (
                   SELECT id FROM classificacoes ORDER BY id DESC LIMIT ?
               )""",
            (ENTRADAS_MAXIMAS,),
        )
    return novo_id


def listar(limite=20):
    """Devolve as classificações mais recentes, da mais nova para a mais antiga."""
    limite = max(1, min(int(limite), ENTRADAS_MAXIMAS))
    with _conectar() as conexao:
        conexao.execute(_ESQUEMA)
        linhas = conexao.execute(
            "SELECT * FROM classificacoes ORDER BY id DESC LIMIT ?", (limite,)
        ).fetchall()

    return [
        {
            "id": l["id"],
            "criado_em": l["criado_em"],
            "usuario_externo": l["usuario_externo"],
            "entradas": json.loads(l["entradas"]),
            "classe_prevista": l["classe_prevista"],
            "probabilidades": json.loads(l["probabilidades"]),
            "indice_risco": l["indice_risco"],
            "intervencao_codigo": l["intervencao_codigo"],
            "avisos": json.loads(l["avisos"]),
            "versao_modelo": l["versao_modelo"],
        }
        for l in linhas
    ]


def total():
    """Quantidade de classificações guardadas no momento."""
    with _conectar() as conexao:
        conexao.execute(_ESQUEMA)
        return conexao.execute("SELECT COUNT(*) FROM classificacoes").fetchone()[0]


def caminho_banco():
    """Caminho do arquivo SQLite em uso."""
    return str(_CAMINHO_BANCO)
