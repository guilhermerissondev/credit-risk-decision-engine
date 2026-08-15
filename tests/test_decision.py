"""Testes da camada de decisão.

A conta de lucro é a parte do projeto que ninguém audita olhando um
gráfico bonito. Se ela estiver errada, o número do README está errado e
todo o resto é decoração.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import decision  # noqa: E402


# --------------------------------------------------------------------------
# Threshold de indiferença
# --------------------------------------------------------------------------
def test_breakeven_formula():
    assert decision.breakeven_pd(margin=0.18, lgd=0.75) == pytest.approx(0.18 / 0.93)


def test_breakeven_com_lgd_alto_fica_mais_conservador():
    """LGD maior significa perda maior por default, logo corte mais duro."""
    assert decision.breakeven_pd(0.20, 0.90) < decision.breakeven_pd(0.20, 0.50)


def test_breakeven_com_margem_alta_fica_mais_permissivo():
    assert decision.breakeven_pd(0.40, 0.75) > decision.breakeven_pd(0.10, 0.75)


def test_no_breakeven_o_valor_esperado_e_zero():
    """Definição: no ponto de indiferença, manter e cortar empatam."""
    m, l = 0.18, 0.75
    t = decision.breakeven_pd(m, l)
    ev = decision.expected_value(np.array([t]), np.array([10_000.0]), m, l)
    assert ev[0] == pytest.approx(0.0, abs=1e-9)


def test_valor_esperado_troca_de_sinal_no_breakeven():
    m, l = 0.18, 0.75
    t = decision.breakeven_pd(m, l)
    exp = np.array([10_000.0])
    assert decision.expected_value(np.array([t - 0.05]), exp, m, l)[0] > 0
    assert decision.expected_value(np.array([t + 0.05]), exp, m, l)[0] < 0


def test_breakeven_independe_da_exposicao():
    """A exposição se cancela na álgebra, então um threshold global é
    ótimo conta a conta. Este teste trava essa propriedade."""
    m, l = 0.18, 0.75
    t = decision.breakeven_pd(m, l)
    for e in (500.0, 10_000.0, 5_000_000.0):
        ev = decision.expected_value(np.array([t]), np.array([e]), m, l)
        assert ev[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Aplicação de política
# --------------------------------------------------------------------------
@pytest.fixture
def carteira():
    pd_hat = np.array([0.05, 0.10, 0.30, 0.60, 0.90])
    exposicao = np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0])
    y = np.array([0, 0, 1, 0, 1])
    return pd_hat, exposicao, y


def test_aprovar_todos_soma_a_carteira_inteira(carteira):
    pd_hat, exp, y = carteira
    r = decision.apply_policy(pd_hat, exp, y, threshold=1.01, margin=0.18, lgd=0.75)
    assert r["contas_aprovadas"] == 5
    assert r["taxa_aprovacao"] == pytest.approx(1.0)

    bons = exp[y == 0].sum() * 0.18
    maus = exp[y == 1].sum() * 0.75
    assert r["lucro_realizado"] == pytest.approx(bons - maus)


def test_recusar_todos_zera_o_resultado(carteira):
    pd_hat, exp, y = carteira
    r = decision.apply_policy(pd_hat, exp, y, threshold=0.0, margin=0.18, lgd=0.75)
    assert r["contas_aprovadas"] == 0
    assert r["lucro_realizado"] == pytest.approx(0.0)
    assert r["lucro_esperado"] == pytest.approx(0.0)


def test_threshold_aprova_estritamente_abaixo(carteira):
    """Contrato do corte: PD < threshold aprova, PD == threshold recusa."""
    pd_hat, exp, y = carteira
    r = decision.apply_policy(pd_hat, exp, y, threshold=0.30, margin=0.18, lgd=0.75)
    assert r["contas_aprovadas"] == 2  # 0.05 e 0.10; o 0.30 fica de fora


def test_receita_e_perda_batem_com_o_lucro(carteira):
    pd_hat, exp, y = carteira
    r = decision.apply_policy(pd_hat, exp, y, threshold=0.70, margin=0.18, lgd=0.75)
    assert r["lucro_realizado"] == pytest.approx(r["receita"] - r["perda"])


def test_aumentar_threshold_nunca_reduz_aprovacao(carteira):
    pd_hat, exp, y = carteira
    taxas = [
        decision.apply_policy(pd_hat, exp, y, t, 0.18, 0.75)["taxa_aprovacao"]
        for t in np.linspace(0, 1, 25)
    ]
    assert all(b >= a for a, b in zip(taxas, taxas[1:]))


# --------------------------------------------------------------------------
# Coerência entre esperado e realizado
# --------------------------------------------------------------------------
def test_com_pd_perfeita_o_esperado_iguala_o_realizado():
    """Se a PD for exatamente o rótulo, as duas contas convergem.

    É o caso-limite que prova que a fórmula do valor esperado e a do
    resultado realizado descrevem a mesma economia.
    """
    y = np.array([0, 1, 0, 1, 0, 0])
    pd_hat = y.astype(float) * (1 - 2e-6) + 1e-6
    exp = np.array([1_000.0, 2_000.0, 3_000.0, 1_500.0, 800.0, 4_000.0])
    r = decision.apply_policy(pd_hat, exp, y, threshold=1.01, margin=0.18, lgd=0.75)
    assert r["lucro_esperado"] == pytest.approx(r["lucro_realizado"], rel=1e-4)


def test_otimo_empirico_bate_o_breakeven_com_pd_perfeita():
    """Com PD perfeita, cortar no break-even remove exatamente os maus."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, size=2000)
    pd_hat = y.astype(float) * 0.99 + 0.005
    exp = rng.uniform(1_000, 20_000, size=2000)

    t = decision.breakeven_pd(0.18, 0.75)
    r = decision.apply_policy(pd_hat, exp, y, t, 0.18, 0.75)
    assert r["inadimplencia_da_carteira"] == pytest.approx(0.0)
    assert r["perda"] == pytest.approx(0.0)


def test_curva_de_lucro_cobre_o_grid():
    rng = np.random.default_rng(1)
    n = 500
    pd_hat = rng.uniform(0, 1, n)
    exp = rng.uniform(1_000, 10_000, n)
    y = rng.binomial(1, pd_hat)
    curva = decision.profit_curve(pd_hat, exp, y)
    assert len(curva) == 200
    assert curva["threshold"].is_monotonic_increasing
    assert {"lucro_realizado", "taxa_aprovacao", "receita", "perda"} <= set(curva.columns)


# --------------------------------------------------------------------------
# Comparação de políticas
# --------------------------------------------------------------------------
def test_percentual_nao_e_calculado_sobre_base_negativa():
    """Guarda contra o bug que inflava o ganho: percentual sobre base
    negativa produz número sem significado."""
    pd_hat = np.array([0.9, 0.95, 0.99])
    exp = np.array([1_000.0, 1_000.0, 1_000.0])
    y = np.array([1, 1, 1])  # carteira só de maus: qualquer aprovação dá prejuízo
    comp = decision.compare_policies(
        pd_hat, exp, y,
        {"Aprovar todos": 1.01, "Threshold 0,50 (default)": 0.50},
        referencia="Threshold 0,50 (default)",
        margin=0.18, lgd=0.75,
    )
    # A referência recusa tudo -> lucro zero -> percentual indefinido.
    assert comp["ganho_pct"].isna().all()


def test_formatacao_de_moeda():
    assert "NT$" in decision.format_currency(1_500_000)
    assert "R$" in decision.format_currency(1_500_000, brl=True)
    assert "mi" in decision.format_currency(4_050_000)
    assert "mil" in decision.format_currency(4_050)
