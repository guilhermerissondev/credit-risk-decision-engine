"""Camada de decisão econômica.

Aqui está o ponto do projeto. Um modelo de PD não decide nada sozinho —
alguém precisa escolher onde cortar. A escolha habitual (0,5 por default,
ou o ponto de máxima F1) não tem relação com o resultado financeiro:
trata errar um bom e errar um mau como se custassem a mesma coisa.

Custam ordens de grandeza diferentes. Aprovar um mau perde LGD x exposição.
Recusar um bom perde margem x exposição. Com margem de 18% e LGD de 75%,
um mau aprovado custa cerca de quatro bons recusados — e é essa razão, não
a curva ROC, que define o corte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Threshold de indiferença
# --------------------------------------------------------------------------
def breakeven_pd(margin: float | None = None, lgd: float | None = None) -> float:
    """PD em que manter e cortar a linha dão o mesmo resultado esperado.

        manter:  (1 - pd) * E * m  -  pd * E * lgd
        cortar:  0

    Igualando e resolvendo, a exposição E se cancela:

        pd* = m / (m + lgd)

    Duas consequências que valem registrar:

    1. O corte ótimo é uma função exclusiva da economia do produto. Não
       depende do modelo, da AUC nem da distribuição de escores.
    2. Como E se cancela, um threshold global é ótimo conta a conta — não
       é preciso um corte por faixa de exposição. Isso deixaria de valer
       se margem ou LGD variassem por segmento.
    """
    m = config.ANNUAL_MARGIN_RATE if margin is None else margin
    l = config.LGD if lgd is None else lgd
    return m / (m + l)


# --------------------------------------------------------------------------
# Resultado de uma política
# --------------------------------------------------------------------------
def expected_value(
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    margin: float | None = None,
    lgd: float | None = None,
) -> np.ndarray:
    """Valor esperado de manter a linha, conta a conta."""
    m = config.ANNUAL_MARGIN_RATE if margin is None else margin
    l = config.LGD if lgd is None else lgd
    return exposure * ((1.0 - pd_hat) * m - pd_hat * l)


def apply_policy(
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    margin: float | None = None,
    lgd: float | None = None,
) -> dict[str, float]:
    """Resultado de aprovar todo mundo com PD abaixo do threshold.

    Reporta lucro ESPERADO (o que o modelo projeta) e lucro REALIZADO (o
    que de fato aconteceu no conjunto avaliado). Os dois só convergem se a
    calibração estiver boa — a diferença entre eles é, na prática, um
    teste de calibração expresso em dinheiro.
    """
    m = config.ANNUAL_MARGIN_RATE if margin is None else margin
    l = config.LGD if lgd is None else lgd

    aprovado = pd_hat < threshold
    n = len(pd_hat)
    n_aprov = int(aprovado.sum())

    exp_aprov = exposure[aprovado]
    y_aprov = y_true[aprovado]

    receita = float((exp_aprov[y_aprov == 0] * m).sum())
    perda = float((exp_aprov[y_aprov == 1] * l).sum())
    realizado = receita - perda

    esperado = float(expected_value(pd_hat[aprovado], exp_aprov, m, l).sum())

    return {
        "threshold": float(threshold),
        "contas_aprovadas": n_aprov,
        "taxa_aprovacao": n_aprov / n if n else 0.0,
        "exposicao_aprovada": float(exp_aprov.sum()),
        "inadimplencia_da_carteira": float(y_aprov.mean()) if n_aprov else 0.0,
        "receita": receita,
        "perda": perda,
        "lucro_realizado": realizado,
        "lucro_esperado": esperado,
        "roa": realizado / exp_aprov.sum() if n_aprov and exp_aprov.sum() else 0.0,
    }


def profit_curve(
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
    margin: float | None = None,
    lgd: float | None = None,
) -> pd.DataFrame:
    """Lucro da carteira ao longo de todos os thresholds possíveis."""
    if grid is None:
        grid = np.linspace(0.01, 1.0, 200)
    rows = [apply_policy(pd_hat, exposure, y_true, t, margin, lgd) for t in grid]
    return pd.DataFrame(rows)


def empirical_optimum(
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    y_true: np.ndarray,
    margin: float | None = None,
    lgd: float | None = None,
) -> float:
    """Threshold que maximiza o lucro observado.

    Deve ser calculado em dado de CALIBRAÇÃO e depois aplicado ao teste.
    Buscar o ótimo no próprio teste e reportar o lucro de lá é escolher e
    avaliar no mesmo dado — o número sai inflado.
    """
    curva = profit_curve(pd_hat, exposure, y_true, margin=margin, lgd=lgd)
    return float(curva.loc[curva["lucro_realizado"].idxmax(), "threshold"])


def max_f1_threshold(pd_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Threshold de máxima F1 — a escolha puramente estatística, aqui
    presente como termo de comparação."""
    from sklearn.metrics import f1_score

    grid = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, (pd_hat >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# --------------------------------------------------------------------------
# Comparação de políticas
# --------------------------------------------------------------------------
def compare_policies(
    pd_test: np.ndarray,
    exposure_test: np.ndarray,
    y_test: np.ndarray,
    policies: dict[str, float],
    referencia: str = "Threshold 0,50 (default)",
    margin: float | None = None,
    lgd: float | None = None,
) -> pd.DataFrame:
    """Tabela final: cada política de corte e o que ela produz em dinheiro.

    A referência para o ganho percentual é o threshold 0,50 — a escolha
    que sai de graça em qualquer `predict()`. Comparar contra "aprovar
    todos" seria conveniente mas desonesto: sem nenhuma política de
    crédito essa carteira é deficitária, então qualquer corte produziria
    um percentual espetacular sobre uma base negativa. O ganho que
    interessa é o que sobra DEPOIS de já se ter um modelo rodando.
    """
    rows = []
    for nome, t in policies.items():
        r = apply_policy(pd_test, exposure_test, y_test, t, margin, lgd)
        r["politica"] = nome
        rows.append(r)

    df = pd.DataFrame(rows).set_index("politica")

    if "Aprovar todos" in df.index:
        df["ganho_vs_sem_politica"] = (
            df["lucro_realizado"] - df.loc["Aprovar todos", "lucro_realizado"]
        )

    if referencia in df.index:
        base = float(df.loc[referencia, "lucro_realizado"])
        df["ganho_vs_referencia"] = df["lucro_realizado"] - base
        # Percentual só faz sentido sobre base positiva.
        df["ganho_pct"] = (
            (df["lucro_realizado"] - base) / base if base > 0 else np.nan
        )
    return df


def format_currency(v: float, brl: bool = False) -> str:
    """Formata em NT$ ou na conversão de referência para BRL."""
    if brl:
        v = v * config.BRL_PER_UNIT
        simbolo = "R$"
    else:
        simbolo = config.CURRENCY
    for limiar, sufixo in ((1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if abs(v) >= limiar:
            return f"{simbolo} {v / limiar:,.2f} {sufixo}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{simbolo} {v:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
