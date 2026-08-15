"""Ingestão do UCI Default of Credit Card Clients.

30.000 contas de cartão de crédito com 6 meses de histórico de fatura,
pagamento e atraso, e a marcação de inadimplência no mês seguinte.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from . import config

# Nomes originais -> nomes legíveis. As colunas PAY_* do UCI são o status
# de pagamento em meses consecutivos: -2 sem uso, -1 pago integral,
# 0 pagou o mínimo/rotativo, 1..9 meses em atraso.
_RENAME = {
    "ID": "id_conta",
    "LIMIT_BAL": "limite_credito",
    "SEX": "sexo",
    "EDUCATION": "escolaridade",
    "MARRIAGE": "estado_civil",
    "AGE": "idade",
    "default payment next month": config.TARGET,
}
for _i, _mes in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], start=1):
    _RENAME[_mes] = f"status_atraso_m{_i}"
for _i in range(1, 7):
    _RENAME[f"BILL_AMT{_i}"] = f"fatura_m{_i}"
    _RENAME[f"PAY_AMT{_i}"] = f"pagamento_m{_i}"


def download(force: bool = False) -> None:
    """Baixa o zip do UCI se ainda não estiver em disco."""
    if config.RAW_ZIP.exists() and not force:
        return
    req = urllib.request.Request(config.UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    config.RAW_ZIP.write_bytes(payload)


def load_raw() -> pd.DataFrame:
    """Lê o .xls de dentro do zip e aplica os nomes legíveis.

    O sufixo _m1 é sempre o mês MAIS RECENTE e _m6 o mais antigo, seguindo
    a convenção do dataset original (BILL_AMT1 = fatura de setembro,
    BILL_AMT6 = abril). Manter isso explícito evita inverter o sinal de
    todas as features de tendência.
    """
    download()
    with zipfile.ZipFile(config.RAW_ZIP) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_excel(io.BytesIO(fh.read()), header=1)

    df = df.rename(columns=_RENAME)

    # Códigos fora do dicionário oficial. O UCI documenta escolaridade
    # 1..4 e estado civil 1..3; o resto (0, 5, 6) aparece no arquivo sem
    # significado definido e é agrupado em "outros".
    df["escolaridade"] = df["escolaridade"].where(df["escolaridade"].between(1, 4), 4)
    df["estado_civil"] = df["estado_civil"].where(df["estado_civil"].between(1, 3), 3)

    return df


def load() -> pd.DataFrame:
    """Dataset pronto para engenharia de features."""
    return load_raw()
