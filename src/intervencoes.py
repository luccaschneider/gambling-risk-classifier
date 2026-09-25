"""
Intervenções estruturadas por nível de risco.

Cada nível devolve um objeto com código estável, lista de ações e parâmetros,
de modo que uma plataforma consiga executá-lo sem interpretar texto livre. A
descrição em português existe para leitura humana, não para ser processada.

As ações seguem o que o trabalho definiu para cada nível:
  Baixo  - notificação informativa sobre tempo de sessão e valores movimentados
  Médio  - limite temporário de depósito e sugestão de pausas entre sessões
  Alto   - suspensão temporária da conta e encaminhamento para suporte especializado
"""

# ---------------------------------------------------------------------------
# Parâmetros numéricos.
#
# Estes são VALORES DE REFERÊNCIA para a demonstração, não recomendação
# clínica. Prazos, percentuais e limiares de intervenção dependem da política
# de jogo responsável da operadora e da regulamentação aplicável, e devem ser
# definidos por ela — se possível com acompanhamento de profissionais da área.
# ---------------------------------------------------------------------------
PARAMETROS_REFERENCIA = {
    "origem": "valores de referência da demonstração, a serem definidos pela operadora",
    "notificacao_informativa": {
        "periodicidade_dias": 30,
        "conteudo": ["tempo_de_sessao", "valores_movimentados"],
    },
    "limite_deposito": {
        "duracao_dias": 7,
        "reducao_percentual": 50,
    },
    "pausa_entre_sessoes": {
        "intervalo_sugerido_minutos": 30,
        "duracao_sugerida_minutos": 15,
    },
    "suspensao_conta": {
        "duracao_dias": 30,
        "reversivel": True,
    },
    "encaminhamento_suporte": {
        "canal": "suporte_jogo_responsavel",
        "prazo_contato_horas": 48,
    },
}


INTERVENCOES = {
    "Baixo risco": {
        "codigo": "MONITORAMENTO_INFORMATIVO",
        "nivel": "Baixo risco",
        "acoes": [
            {
                "tipo": "ENVIAR_NOTIFICACAO",
                "alvo": "usuario",
                "conteudo": "resumo_atividade",
                "bloqueia_conta": False,
                "parametros": PARAMETROS_REFERENCIA["notificacao_informativa"],
            },
        ],
        "descricao": (
            "Envio de notificação informativa sobre tempo de sessão e valores "
            "movimentados. Mantém o monitoramento padrão, sem restrição de uso."
        ),
    },
    "Medio risco": {
        "codigo": "LIMITE_TEMPORARIO",
        "nivel": "Médio risco",
        "acoes": [
            {
                "tipo": "APLICAR_LIMITE_DEPOSITO",
                "alvo": "conta",
                "carater": "temporario",
                "bloqueia_conta": False,
                "parametros": PARAMETROS_REFERENCIA["limite_deposito"],
            },
            {
                "tipo": "SUGERIR_PAUSA",
                "alvo": "usuario",
                "carater": "sugestao",
                "bloqueia_conta": False,
                "parametros": PARAMETROS_REFERENCIA["pausa_entre_sessoes"],
            },
        ],
        "descricao": (
            "Aplicação de limite temporário de depósito e sugestão de pausas "
            "entre sessões. A conta permanece ativa."
        ),
    },
    "Alto risco": {
        "codigo": "SUSPENSAO_TEMPORARIA",
        "nivel": "Alto risco",
        "acoes": [
            {
                "tipo": "SUSPENDER_CONTA",
                "alvo": "conta",
                "carater": "temporario",
                "bloqueia_conta": True,
                "parametros": PARAMETROS_REFERENCIA["suspensao_conta"],
            },
            {
                "tipo": "ENCAMINHAR_SUPORTE",
                "alvo": "equipe_interna",
                "carater": "atendimento",
                "bloqueia_conta": False,
                "parametros": PARAMETROS_REFERENCIA["encaminhamento_suporte"],
            },
        ],
        "descricao": (
            "Suspensão temporária da conta e encaminhamento para canal de "
            "suporte especializado em jogo responsável."
        ),
    },
}


def intervencao_para(classe: str) -> dict:
    """Devolve a intervenção estruturada correspondente à classe de risco."""
    return INTERVENCOES[classe]
