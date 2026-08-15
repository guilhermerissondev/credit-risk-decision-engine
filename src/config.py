"""Parâmetros centrais do motor de decisão.

Toda a economia do projeto vive aqui. Mudar uma premissa de negócio
não deve exigir tocar em nenhum outro módulo.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_ZIP = DATA_RAW / "uci.zip"
TARGET = "inadimplente"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# --------------------------------------------------------------------------
# Moeda
#
# O dataset é de um emissor de cartões de Taiwan e os valores estão em
# novos dólares taiwaneses (NT$). Reportamos em NT$ e oferecemos uma
# conversão explícita para BRL — com a taxa à vista, para que ninguém
# confunda o número convertido com precisão que ele não tem.
# --------------------------------------------------------------------------
CURRENCY = "NT$"
BRL_PER_UNIT = 0.18  # câmbio de referência NT$ -> R$; ajuste conforme necessário

# --------------------------------------------------------------------------
# Premissas econômicas da decisão
#
# A decisão modelada NÃO é originação de crédito novo — é gestão de conta:
# manter/renovar a linha rotativa de um cliente existente ou cortá-la.
# É o que o dado suporta, porque temos 6 meses de histórico de fatura e
# pagamento de contas já abertas.
#
#   Manter a linha  -> ganha margem sobre a exposição se o cliente pagar,
#                      perde LGD x exposição se ele quebrar.
#   Cortar a linha  -> resultado zero (nem margem, nem perda).
# --------------------------------------------------------------------------

# Margem líquida anual sobre o saldo rotativo médio: receita de juros e
# intercâmbio menos custo de funding e operação.
ANNUAL_MARGIN_RATE = 0.18

# Loss Given Default: fração da exposição efetivamente perdida após
# recuperação, venda de carteira e cobrança.
LGD = 0.75

# Piso de exposição. Contas com fatura atual muito baixa geram pouca
# receita; sem um piso, o modelo econômico as trata como irrelevantes.
MIN_EXPOSURE = 1_000.0

# --------------------------------------------------------------------------
# Atributos protegidos
#
# SEX e MARRIAGE existem no dataset original e são preditivos. Eles são
# DELIBERADAMENTE excluídos das features: usar sexo ou estado civil em
# decisão de crédito é discriminação vedada (no Brasil, CDC art. 39 e a
# Res. CMN 4.949; nos EUA, ECOA/Reg B). Eles ficam retidos apenas como
# eixo de auditoria de fairness sobre o resultado do modelo.
# --------------------------------------------------------------------------
PROTECTED_ATTRIBUTES = ["sexo", "estado_civil", "faixa_etaria"]
