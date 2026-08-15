"""Engenharia de features com semântica de risco.

O UCI entrega colunas cifradas (PAY_0, BILL_AMT3...). Um modelo treinado
direto nelas atinge boa AUC e não diz nada a um analista de crédito.
Aqui elas viram grandezas que um comitê de risco reconhece: utilização
de limite, taxa de pagamento, severidade e tendência de atraso.

Convenção herdada do dataset: _m1 é o mês mais recente, _m6 o mais antigo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MESES = [1, 2, 3, 4, 5, 6]
_EPS = 1e-9


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Razão protegida contra divisão por zero e denominador negativo.

    Fatura negativa significa saldo credor (o cliente pagou a mais). Para
    razões de utilização e pagamento isso não tem interpretação útil, então
    o denominador é pisado em zero e o resultado vira 0.
    """
    den = den.clip(lower=0)
    return (num / (den + _EPS)).where(den > 0, 0.0)


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Constrói a matriz de features.

    Retorna
    -------
    X : features do modelo (sem atributos protegidos)
    y : alvo binário de inadimplência
    aux : exposição e atributos protegidos, para a camada econômica e
          para a auditoria de fairness — nunca para o treino
    """
    out = pd.DataFrame(index=df.index)

    limite = df["limite_credito"].clip(lower=1.0)
    faturas = df[[f"fatura_m{m}" for m in MESES]]
    pagamentos = df[[f"pagamento_m{m}" for m in MESES]]
    atrasos = df[[f"status_atraso_m{m}" for m in MESES]]

    # ---- Capacidade e exposição -----------------------------------------
    out["limite_credito"] = df["limite_credito"]
    out["idade"] = df["idade"]
    # Mantida como inteiro, não como dtype `category` do pandas: o LightGBM
    # guarda o mapeamento de categorias do treino e recusa a predição se ele
    # não bater exatamente. Um round-trip por parquet é suficiente para
    # quebrar esse acoplamento, e o modelo só falharia na hora de servir.
    # A natureza categórica é declarada ao LightGBM em model.CATEGORICAL.
    out["escolaridade"] = df["escolaridade"].astype(int)

    # ---- Utilização de limite -------------------------------------------
    # O quanto da linha o cliente já consumiu. Utilização alta e crescente
    # é o sinal clássico de aperto de liquidez.
    utilizacao = pd.DataFrame(
        {m: _safe_ratio(faturas[f"fatura_m{m}"], limite) for m in MESES}
    )
    out["utilizacao_atual"] = utilizacao[1]
    out["utilizacao_media_6m"] = utilizacao.mean(axis=1)
    out["utilizacao_max_6m"] = utilizacao.max(axis=1)
    out["tendencia_utilizacao"] = utilizacao[1] - utilizacao[6]
    out["volatilidade_utilizacao"] = utilizacao.std(axis=1)
    out["folga_limite"] = df["limite_credito"] - faturas["fatura_m1"]

    # ---- Comportamento de pagamento -------------------------------------
    # PAY_AMT{t} quita a fatura de BILL_AMT{t+1}: o pagamento feito no mês t
    # se refere ao que foi faturado no mês anterior. Alinhar isso importa —
    # dividir pagamento e fatura do mesmo índice mede a coisa errada.
    taxas = pd.DataFrame(
        {
            t: _safe_ratio(pagamentos[f"pagamento_m{t}"], faturas[f"fatura_m{t + 1}"])
            for t in range(1, 6)
        }
    )
    taxas = taxas.clip(upper=2.0)  # pagamentos muito acima da fatura viram outliers

    out["taxa_pagamento_atual"] = taxas[1]
    out["taxa_pagamento_media_5m"] = taxas.mean(axis=1)
    out["taxa_pagamento_min_5m"] = taxas.min(axis=1)
    out["tendencia_taxa_pagamento"] = taxas[1] - taxas[5]
    out["meses_sem_pagamento_6m"] = (pagamentos <= 0).sum(axis=1)
    out["meses_quitacao_total_5m"] = (taxas >= 0.98).sum(axis=1)
    out["pagamento_total_6m"] = pagamentos.sum(axis=1)
    out["fatura_media_6m"] = faturas.mean(axis=1)

    # ---- Severidade e trajetória de atraso ------------------------------
    out["atraso_mes_atual"] = atrasos["status_atraso_m1"]
    out["pior_atraso_6m"] = atrasos.max(axis=1)
    out["meses_em_atraso_6m"] = (atrasos >= 1).sum(axis=1)
    out["meses_atraso_grave_6m"] = (atrasos >= 2).sum(axis=1)
    out["tendencia_atraso"] = (
        atrasos["status_atraso_m1"] - atrasos["status_atraso_m6"]
    )
    out["nunca_usou_6m"] = (atrasos <= -2).all(axis=1).astype(int)

    # ---- Sinal composto de deterioração ---------------------------------
    # Saldo subindo enquanto o pagamento cai: o cliente está financiando
    # consumo com a linha em vez de amortizá-la.
    out["saldo_cresce_pagamento_cai"] = (
        (out["tendencia_utilizacao"] > 0) & (out["tendencia_taxa_pagamento"] < 0)
    ).astype(int)

    y = df[config.TARGET].astype(int)

    # ---- Auxiliares: economia e auditoria -------------------------------
    aux = pd.DataFrame(index=df.index)
    # Exposure at Default: o saldo em aberto hoje é o que fica exposto se a
    # conta quebrar no mês seguinte.
    aux["exposicao"] = faturas["fatura_m1"].clip(lower=0).clip(lower=config.MIN_EXPOSURE)
    aux["sexo"] = df["sexo"].map({1: "masculino", 2: "feminino"})
    aux["estado_civil"] = df["estado_civil"].map(
        {1: "casado", 2: "solteiro", 3: "outros"}
    )
    aux["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=["ate_25", "26_35", "36_45", "46_55", "56_mais"],
    )

    return out, y, aux


def feature_names(X: pd.DataFrame) -> list[str]:
    return list(X.columns)
