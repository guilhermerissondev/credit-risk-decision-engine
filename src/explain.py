"""Reason codes: por que esta conta foi recusada.

Crédito é domínio regulado. Negar uma linha sem saber apontar o motivo
é problema jurídico antes de ser problema técnico — e um modelo que não
explica a decisão individual não passa em validação.

Para modelos de árvore não é preciso a biblioteca `shap`: o próprio
LightGBM devolve os valores SHAP exatos via `pred_contrib=True`, usando
o TreeSHAP interno. Uma dependência pesada a menos e o resultado é o
mesmo — os valores são exatos, não aproximados por amostragem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tradução de feature para linguagem de carta de recusa.
_MOTIVOS = {
    "atraso_mes_atual": "atraso registrado no ciclo mais recente",
    "pior_atraso_6m": "severidade do pior atraso nos últimos 6 meses",
    "meses_em_atraso_6m": "quantidade de meses com atraso no semestre",
    "meses_atraso_grave_6m": "meses com atraso igual ou superior a 2 ciclos",
    "tendencia_atraso": "piora na trajetória de atraso ao longo do semestre",
    "utilizacao_atual": "utilização do limite acima do usual",
    "utilizacao_media_6m": "utilização média do limite no semestre",
    "utilizacao_max_6m": "pico de utilização do limite no semestre",
    "tendencia_utilizacao": "crescimento do saldo utilizado",
    "volatilidade_utilizacao": "oscilação atípica no uso do limite",
    "folga_limite": "margem disponível no limite",
    "taxa_pagamento_atual": "parcela da fatura quitada no último ciclo",
    "taxa_pagamento_media_5m": "percentual médio da fatura pago no semestre",
    "taxa_pagamento_min_5m": "pior percentual de pagamento do semestre",
    "tendencia_taxa_pagamento": "queda no percentual da fatura sendo pago",
    "meses_sem_pagamento_6m": "meses sem qualquer pagamento registrado",
    "meses_quitacao_total_5m": "meses com quitação integral da fatura",
    "pagamento_total_6m": "volume total pago no semestre",
    "fatura_media_6m": "valor médio da fatura no semestre",
    "saldo_cresce_pagamento_cai": "saldo crescendo enquanto o pagamento diminui",
    "limite_credito": "limite de crédito concedido",
    "idade": "idade do titular",
    "escolaridade": "escolaridade informada",
    "nunca_usou_6m": "ausência de movimentação no semestre",
}


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Contribuições SHAP exatas por feature, em log-odds.

    A última coluna devolvida pelo LightGBM é o valor base (a expectativa
    do modelo antes de olhar qualquer feature) e é descartada aqui.
    """
    raw = model.booster.predict(X[model.features], pred_contrib=True)
    raw = np.asarray(raw)
    return pd.DataFrame(raw[:, :-1], columns=model.features, index=X.index)


def base_value(model, X: pd.DataFrame) -> float:
    raw = np.asarray(model.booster.predict(X[model.features].head(1), pred_contrib=True))
    return float(raw[0, -1])


def reason_codes(
    model,
    X: pd.DataFrame,
    top_n: int = 4,
) -> list[list[dict]]:
    """Para cada conta, os fatores que mais empurraram o escore para risco.

    Só contribuições positivas entram: em uma recusa, o que interessa
    justificar é o que pesou contra, não o que pesou a favor.
    """
    contrib = contributions(model, X)
    saida: list[list[dict]] = []

    for _, linha in contrib.iterrows():
        positivos = linha[linha > 0].sort_values(ascending=False).head(top_n)
        saida.append(
            [
                {
                    "feature": nome,
                    "motivo": _MOTIVOS.get(nome, nome.replace("_", " ")),
                    "impacto_log_odds": float(valor),
                }
                for nome, valor in positivos.items()
            ]
        )
    return saida


def explain_one(model, X: pd.DataFrame, i: int, top_n: int = 5) -> pd.DataFrame:
    """Explicação completa de uma conta, positivos e negativos."""
    contrib = contributions(model, X.iloc[[i]]).iloc[0]
    ordenado = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return pd.DataFrame(
        {
            "fator": [_MOTIVOS.get(k, k) for k in ordenado.index],
            "valor_da_conta": [X.iloc[i][k] for k in ordenado.index],
            "impacto": ordenado.to_numpy(),
            "direcao": ["aumenta o risco" if v > 0 else "reduz o risco" for v in ordenado],
        }
    )


def global_importance(model, X: pd.DataFrame, sample: int = 5000) -> pd.DataFrame:
    """Importância global como média do |SHAP| — mede efeito no escore,
    não quantas vezes a feature foi usada em um split."""
    amostra = X.sample(min(sample, len(X)), random_state=0)
    contrib = contributions(model, amostra)
    imp = contrib.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "impacto_medio": imp.to_numpy()})
