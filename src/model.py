"""Modelo de PD com calibração de probabilidade verificada.

Ordenar risco e estimar risco são problemas diferentes. LightGBM ordena
bem (AUC, KS), mas nada garante que o escore que ele devolve seja uma
probabilidade utilizável.

Para a camada de decisão a diferença é fatal: o cálculo de lucro esperado
multiplica PD por dinheiro. Se a PD estiver enviesada, a curva de lucro é
ficção e o threshold "ótimo" é ruído.

O que este módulo NÃO faz é assumir que calibrar ajuda. Três candidatos
— escore bruto, isotônico e Platt — competem em um conjunto de seleção
separado, e vence o de menor Brier. Em dado desbalanceado com peso de
classe padrão, o escore bruto frequentemente já está calibrado, e nesse
caso a etapa correta é não fazer nada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from . import config

# Features tratadas como categóricas pelo LightGBM. Declaradas por nome em
# vez de por dtype do pandas — ver o comentário em features.build().
CATEGORICAL = ["escolaridade"]

LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS: separação máxima entre as acumuladas de bons e maus.

    É a métrica que times de crédito acompanham no dia a dia, mais que AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "ks": ks_statistic(y_true, p),
        "brier": float(brier_score_loss(y_true, p)),
        "pd_media_prevista": float(np.mean(p)),
        "taxa_observada": float(np.mean(y_true)),
    }


# --------------------------------------------------------------------------
# Calibradores
# --------------------------------------------------------------------------
class IdentityCalibrator:
    """Não faz nada, de propósito.

    Existe para que "não calibrar" seja um candidato explícito na seleção
    em vez de uma omissão.
    """

    name = "bruto"

    def fit(self, p, y):  # noqa: D102
        return self

    def predict(self, p):  # noqa: D102
        return np.asarray(p, dtype=float)


class IsotonicCalibrator:
    name = "isotonico"

    def __init__(self):
        self._m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p, y):
        self._m.fit(p, y)
        return self

    def predict(self, p):
        return self._m.predict(p)


class PlattCalibrator:
    name = "platt"

    def __init__(self):
        self._m = LogisticRegression(C=1e6)

    def fit(self, p, y):
        self._m.fit(np.asarray(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self._m.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


# --------------------------------------------------------------------------
# Modelo servido
# --------------------------------------------------------------------------
@dataclass
class CalibratedRiskModel:
    booster: LGBMClassifier
    calibrator: object
    features: list[str]
    calibrator_name: str = "bruto"
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X[self.features])[:, 1]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade de default pronta para entrar na conta de dinheiro."""
        return np.clip(self.calibrator.predict(self.raw_score(X)), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------
# Partições
# --------------------------------------------------------------------------
def split(X: pd.DataFrame, y: pd.Series, aux: pd.DataFrame):
    """Quatro conjuntos, cada um com uma função única.

        treino     ajusta as árvores
        early-stop define o número de árvores
        calibração ajusta e ESCOLHE o calibrador
        teste      só é tocado uma vez, no fim

    A separação entre early-stop e calibração não é preciosismo: se o
    mesmo conjunto parasse o boosting e ajustasse o calibrador, o
    calibrador estaria corrigindo um viés que ele ajudou a criar.
    """
    idx = np.arange(len(X))
    idx_dev, idx_test = train_test_split(
        idx, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    idx_fit, idx_calib = train_test_split(
        idx_dev, test_size=0.20, stratify=y.iloc[idx_dev], random_state=config.RANDOM_STATE
    )
    idx_train, idx_es = train_test_split(
        idx_fit, test_size=0.15, stratify=y.iloc[idx_fit], random_state=config.RANDOM_STATE
    )

    def take(i):
        return X.iloc[i], y.iloc[i], aux.iloc[i]

    return take(idx_train), take(idx_es), take(idx_calib), take(idx_test)


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
) -> tuple[CalibratedRiskModel, dict]:
    """Treina o booster e seleciona o calibrador por Brier fora da amostra."""
    booster = LGBMClassifier(**LGBM_PARAMS)
    booster.fit(
        X_train,
        y_train,
        eval_X=X_es,
        eval_y=y_es,
        eval_metric="auc",
        categorical_feature=[c for c in CATEGORICAL if c in X_train.columns],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    # A calibração se divide de novo: metade ajusta os calibradores, a
    # outra metade decide qual deles vence. Escolher no mesmo dado do
    # ajuste elegeria sempre o isotônico, que é o mais flexível.
    i_fit, i_sel = train_test_split(
        np.arange(len(X_calib)),
        test_size=0.5,
        stratify=y_calib,
        random_state=config.RANDOM_STATE,
    )
    p_fit = booster.predict_proba(X_calib.iloc[i_fit])[:, 1]
    p_sel = booster.predict_proba(X_calib.iloc[i_sel])[:, 1]
    y_fit = y_calib.iloc[i_fit].to_numpy()
    y_sel = y_calib.iloc[i_sel].to_numpy()

    candidatos = [IdentityCalibrator(), IsotonicCalibrator(), PlattCalibrator()]
    linhas = []
    for c in candidatos:
        c.fit(p_fit, y_fit)
        p_hat = np.clip(c.predict(p_sel), 1e-6, 1 - 1e-6)
        linhas.append(
            {
                "calibrador": c.name,
                "brier_selecao": float(brier_score_loss(y_sel, p_hat)),
                "auc_selecao": float(roc_auc_score(y_sel, p_hat)),
                "pd_media": float(p_hat.mean()),
                "taxa_observada": float(y_sel.mean()),
            }
        )

    selecao = pd.DataFrame(linhas).sort_values("brier_selecao").reset_index(drop=True)
    vencedor_nome = selecao.loc[0, "calibrador"]
    vencedor = next(c for c in candidatos if c.name == vencedor_nome)

    model = CalibratedRiskModel(
        booster=booster,
        calibrator=vencedor,
        features=list(X_train.columns),
        calibrator_name=vencedor_nome,
        selection=selecao,
    )

    info = {
        "melhor_iteracao": int(booster.best_iteration_ or LGBM_PARAMS["n_estimators"]),
        "selecao_calibrador": selecao,
    }
    return model, info


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """PD média prevista x taxa observada por decil de risco.

    É a tabela que o validador de modelo pede: se as duas colunas não
    baterem, a economia construída em cima não vale.
    """
    df = pd.DataFrame({"p": p, "y": y_true})
    df["bin"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
    tab = (
        df.groupby("bin", observed=True)
        .agg(
            contas=("y", "size"),
            pd_prevista=("p", "mean"),
            taxa_observada=("y", "mean"),
        )
        .reset_index(drop=True)
    )
    tab["desvio"] = tab["pd_prevista"] - tab["taxa_observada"]
    return tab
