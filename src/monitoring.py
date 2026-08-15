"""Monitoramento de estabilidade: PSI e CSI.

Modelo de crédito não morre de erro de código — morre de mudança de
população. A safra que entra em 2027 não é a de 2026, e o corte
calibrado hoje aprova outra coisa amanhã sem ninguém perceber.

PSI (Population Stability Index) sobre o escore e CSI (Characteristic
Stability Index) sobre cada feature são o instrumento padrão. Leitura
convencional em risco:

    PSI < 0,10   população estável
    0,10 a 0,25  deslocamento moderado, investigar
    PSI > 0,25   deslocamento relevante, recalibrar ou retreinar
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def psi(
    esperado: np.ndarray,
    observado: np.ndarray,
    n_bins: int = 10,
) -> float:
    """PSI entre uma distribuição de referência e uma nova.

    Os cortes vêm sempre dos quantis da referência: reparticionar com os
    quantis da nova amostra esconderia exatamente o deslocamento que se
    quer medir.
    """
    esperado = np.asarray(esperado, dtype=float)
    observado = np.asarray(observado, dtype=float)

    cortes = np.quantile(esperado, np.linspace(0, 1, n_bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    cortes = np.unique(cortes)

    exp_pct = np.histogram(esperado, bins=cortes)[0] / max(len(esperado), 1)
    obs_pct = np.histogram(observado, bins=cortes)[0] / max(len(observado), 1)

    exp_pct = np.clip(exp_pct, _EPS, None)
    obs_pct = np.clip(obs_pct, _EPS, None)

    return float(np.sum((obs_pct - exp_pct) * np.log(obs_pct / exp_pct)))


def classify_psi(valor: float) -> str:
    if valor < 0.10:
        return "estavel"
    if valor < 0.25:
        return "moderado"
    return "relevante"


def csi_table(
    X_ref: pd.DataFrame,
    X_novo: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """PSI feature a feature, para localizar de onde vem o deslocamento."""
    linhas = []
    for col in X_ref.columns:
        if not pd.api.types.is_numeric_dtype(X_ref[col]):
            continue
        valor = psi(X_ref[col].to_numpy(), X_novo[col].to_numpy(), n_bins=n_bins)
        linhas.append({"feature": col, "csi": valor, "leitura": classify_psi(valor)})
    return pd.DataFrame(linhas).sort_values("csi", ascending=False).reset_index(drop=True)


def simulate_drift(
    X: pd.DataFrame,
    fracao: float = 0.30,
    random_state: int = 0,
) -> pd.DataFrame:
    """Gera uma safra artificialmente deteriorada.

    O cenário simulado é o que de fato acontece em virada de ciclo
    econômico: uma parcela da base passa a atrasar mais e a usar mais
    limite, sem que nada no modelo tenha mudado.

    Serve para provar que o monitor acusa quando deveria. Um PSI que
    nunca disparou não é evidência de estabilidade — é ausência de teste,
    e foi assim que a primeira versão deste projeto passou batido por um
    deslocamento que o alarme deveria ter pego.
    """
    rng = np.random.default_rng(random_state)
    novo = X.copy()
    n = len(novo)
    atingidos = rng.choice(n, size=int(n * fracao), replace=False)
    pos = novo.index[atingidos]

    # Atraso é a família de features que mais move o escore; deslocar
    # utilização sozinha não produz efeito mensurável no PSI do escore.
    for col, delta in (
        ("atraso_mes_atual", 1),
        ("pior_atraso_6m", 1),
        ("meses_em_atraso_6m", 1),
    ):
        if col in novo.columns:
            novo.loc[pos, col] = novo.loc[pos, col] + delta

    for col, delta in (
        ("utilizacao_atual", 0.25),
        ("utilizacao_media_6m", 0.20),
        ("tendencia_utilizacao", 0.15),
    ):
        if col in novo.columns:
            novo.loc[pos, col] = novo.loc[pos, col] + delta

    if "taxa_pagamento_media_5m" in novo.columns:
        novo.loc[pos, "taxa_pagamento_media_5m"] = (
            novo.loc[pos, "taxa_pagamento_media_5m"] * 0.5
        )

    return novo


def fairness_report(
    aux: pd.DataFrame,
    pd_hat: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    atributos: list[str],
) -> pd.DataFrame:
    """Taxa de aprovação e inadimplência por grupo protegido.

    Sexo e estado civil ficaram fora das features de propósito, mas
    excluir a variável não garante ausência de efeito: outras features
    podem funcionar como proxy. A verificação é sobre o RESULTADO —
    se a taxa de aprovação diverge muito entre grupos com risco
    observado parecido, há disparidade a explicar.
    """
    aprovado = pd_hat < threshold
    linhas = []
    for attr in atributos:
        if attr not in aux.columns:
            continue
        grupos = aux[attr].astype("object").fillna("nao_informado")
        for grupo, mascara in grupos.groupby(grupos).groups.items():
            sel = aux.index.isin(mascara)
            if sel.sum() < 30:
                continue
            linhas.append(
                {
                    "atributo": attr,
                    "grupo": grupo,
                    "contas": int(sel.sum()),
                    "taxa_aprovacao": float(aprovado[sel].mean()),
                    "pd_media": float(pd_hat[sel].mean()),
                    "inadimplencia_observada": float(y_true[sel].mean()),
                }
            )
    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    # Disparate impact: razão entre a taxa de aprovação do grupo e a do
    # grupo mais aprovado dentro do mesmo atributo.
    df["indice_paridade"] = df.groupby("atributo")["taxa_aprovacao"].transform(
        lambda s: s / s.max()
    )
    return df
