"""Simulador de política de crédito.

Duas telas: uma para o gestor da carteira decidir onde cortar, outra
para o analista entender uma conta específica.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, decision, explain  # noqa: E402

st.set_page_config(page_title="Motor de Decisão de Crédito", page_icon="▩", layout="wide")

AZUL, VERMELHO, VERDE, CINZA = "#1f4e79", "#c0392b", "#27ae60", "#7f8c8d"


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    caminho = config.MODELS / "modelo_calibrado.joblib"
    if not caminho.exists():
        return None
    return joblib.load(caminho)


@st.cache_data
def carregar_carteira():
    caminho = config.DATA_PROCESSED / "teste_scorado.parquet"
    if not caminho.exists():
        return None
    return pd.read_parquet(caminho)


modelo = carregar_modelo()
carteira = carregar_carteira()

if modelo is None or carteira is None:
    st.error(
        "Artefatos não encontrados. Rode o pipeline antes de abrir o app:\n\n"
        "```\npython -m scripts.train\n```"
    )
    st.stop()

pd_hat = carteira["pd"].to_numpy()
exposicao = carteira["exposicao"].to_numpy()
y_true = carteira[config.TARGET].to_numpy()
X = carteira[modelo.features]


# --------------------------------------------------------------------------
# Premissas
# --------------------------------------------------------------------------
st.sidebar.header("Premissas econômicas")
st.sidebar.caption(
    "A decisão modelada é de gestão de conta: manter ou cortar a linha "
    "rotativa de um cliente existente."
)

margem = st.sidebar.slider(
    "Margem líquida anual sobre a exposição", 0.02, 0.60,
    float(config.ANNUAL_MARGIN_RATE), 0.01,
    help="Juros e intercâmbio menos custo de funding e operação.",
)
lgd = st.sidebar.slider(
    "LGD — perda dado o default", 0.10, 1.00, float(config.LGD), 0.05,
    help="Fração da exposição perdida após recuperação e cobrança.",
)

t_breakeven = decision.breakeven_pd(margem, lgd)

st.sidebar.markdown("---")
st.sidebar.metric("Threshold de indiferença", f"{t_breakeven:.1%}")
st.sidebar.caption(
    f"pd* = m / (m + LGD) = {margem:.2f} / ({margem:.2f} + {lgd:.2f}) = {t_breakeven:.4f}\n\n"
    "A exposição se cancela na álgebra, então este corte é ótimo conta a conta."
)
st.sidebar.markdown("---")
em_brl = st.sidebar.checkbox("Converter para BRL", value=False)
st.sidebar.caption(
    f"Valores originais em {config.CURRENCY}. Conversão de referência a "
    f"R$ {config.BRL_PER_UNIT} por unidade — não use para nada além de leitura."
)


def moeda(v: float) -> str:
    return decision.format_currency(v, brl=em_brl)


# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.title("Motor de decisão de crédito")
st.caption(
    f"{len(carteira):,} contas em carteira · modelo servido: {modelo.calibrator_name} · "
    "o corte não vem da curva ROC, vem da economia do produto".replace(",", ".")
)

aba_politica, aba_conta = st.tabs(["Política da carteira", "Análise de conta"])


# ==========================================================================
# Política
# ==========================================================================
with aba_politica:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Ponto de corte")
        usar_be = st.checkbox("Usar o break-even econômico", value=True)
        threshold = (
            t_breakeven
            if usar_be
            else st.slider("Threshold de PD (aprova abaixo)", 0.01, 1.0, 0.50, 0.005)
        )
        if usar_be:
            st.slider(
                "Threshold de PD (aprova abaixo)", 0.01, 1.0, float(t_breakeven), 0.005,
                disabled=True,
            )

    r = decision.apply_policy(pd_hat, exposicao, y_true, threshold, margem, lgd)
    r_050 = decision.apply_policy(pd_hat, exposicao, y_true, 0.50, margem, lgd)
    r_todos = decision.apply_policy(pd_hat, exposicao, y_true, 1.01, margem, lgd)

    with col_b:
        st.subheader("Resultado da política")
        m1, m2, m3 = st.columns(3)
        m1.metric("Taxa de aprovação", f"{r['taxa_aprovacao']:.1%}")
        m2.metric("Inadimplência da carteira", f"{r['inadimplencia_da_carteira']:.2%}",
                  delta=f"{r['inadimplencia_da_carteira'] - r_todos['inadimplencia_da_carteira']:.2%}",
                  delta_color="inverse")
        m3.metric("Lucro", moeda(r["lucro_realizado"]),
                  delta=moeda(r["lucro_realizado"] - r_050["lucro_realizado"]) + " vs corte 0,50")

    st.markdown("---")

    curva = decision.profit_curve(pd_hat, exposicao, y_true, margin=margem, lgd=lgd)
    idx_max = curva["lucro_realizado"].idxmax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curva["threshold"], y=curva["lucro_realizado"],
        mode="lines", name="lucro da carteira",
        line=dict(color=AZUL, width=2.5),
        hovertemplate="corte %{x:.3f}<br>lucro %{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=CINZA, width=1))
    fig.add_vline(x=threshold, line=dict(color=VERMELHO, width=2, dash="dash"),
                  annotation_text=f"política atual ({threshold:.3f})")
    fig.add_vline(x=float(curva.loc[idx_max, "threshold"]),
                  line=dict(color=VERDE, width=1.5, dash="dot"),
                  annotation_text="máximo observado", annotation_position="top left")
    fig.add_vline(x=0.5, line=dict(color=CINZA, width=1, dash="dashdot"),
                  annotation_text="0,50", annotation_position="bottom right")
    fig.update_layout(
        title="Lucro da carteira por ponto de corte",
        xaxis_title="Threshold de PD (aprova abaixo)",
        yaxis_title=f"Lucro ({'R$' if em_brl else config.CURRENCY})",
        height=420, margin=dict(t=50, b=40), showlegend=False,
    )
    if em_brl:
        fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Comparação de políticas")
    politicas = {
        "Aprovar todos": 1.01,
        "Threshold 0,50 (default)": 0.50,
        "Break-even econômico": t_breakeven,
        "Política atual": threshold,
    }
    comp = decision.compare_policies(
        pd_hat, exposicao, y_true, politicas,
        referencia="Threshold 0,50 (default)", margin=margem, lgd=lgd,
    )
    tabela = pd.DataFrame({
        "Corte": comp["threshold"].map(lambda v: f"{v:.3f}"),
        "Aprovação": comp["taxa_aprovacao"].map(lambda v: f"{v:.1%}"),
        "Inadimplência": comp["inadimplencia_da_carteira"].map(lambda v: f"{v:.2%}"),
        "Receita": comp["receita"].map(moeda),
        "Perda": comp["perda"].map(moeda),
        "Lucro": comp["lucro_realizado"].map(moeda),
        "vs corte 0,50": comp["ganho_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1%}"),
    })
    st.dataframe(tabela, width='stretch')

    if r_todos["lucro_realizado"] < 0:
        st.info(
            f"Sem nenhuma política de crédito esta carteira perde "
            f"{moeda(abs(r_todos['lucro_realizado']))}: com inadimplência de "
            f"{r_todos['inadimplencia_da_carteira']:.1%} e margem de {margem:.0%}, "
            f"a receita dos bons não cobre a perda dos maus. É por isso que existe "
            f"política de crédito — e por que o ganho relevante se mede contra o "
            f"corte padrão de 0,50, não contra o cenário sem política."
        )


# ==========================================================================
# Conta individual
# ==========================================================================
with aba_conta:
    st.subheader("Análise de conta")

    col1, col2 = st.columns([1, 1])
    with col1:
        filtro = st.radio(
            "Selecionar entre", ["Todas", "Recusadas na política atual", "Aprovadas"],
            horizontal=True,
        )
    aprovado_mask = pd_hat < threshold
    if filtro == "Recusadas na política atual":
        universo = np.where(~aprovado_mask)[0]
    elif filtro == "Aprovadas":
        universo = np.where(aprovado_mask)[0]
    else:
        universo = np.arange(len(carteira))

    if len(universo) == 0:
        st.warning("Nenhuma conta neste filtro com o corte atual.")
        st.stop()

    with col2:
        posicao = st.number_input(
            f"Conta ({len(universo):,} disponíveis)".replace(",", "."),
            min_value=0, max_value=len(universo) - 1, value=0, step=1,
        )
    i = int(universo[int(posicao)])

    pd_conta = float(pd_hat[i])
    aprovada = pd_conta < threshold
    ev = float(decision.expected_value(
        np.array([pd_conta]), np.array([exposicao[i]]), margem, lgd
    )[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PD estimada", f"{pd_conta:.1%}")
    c2.metric("Exposição", moeda(float(exposicao[i])))
    c3.metric("Valor esperado", moeda(ev))
    c4.metric("Decisão", "Manter linha" if aprovada else "Cortar linha")

    if aprovada:
        st.success(
            f"PD de {pd_conta:.1%} está abaixo do corte de {threshold:.1%}. "
            f"Manter a linha rende {moeda(ev)} em valor esperado."
        )
    else:
        st.error(
            f"PD de {pd_conta:.1%} está acima do corte de {threshold:.1%}. "
            f"Manter a linha custaria {moeda(ev)} em valor esperado."
        )

    st.markdown("#### Fatores da decisão")
    st.caption(
        "Valores SHAP exatos, calculados pelo TreeSHAP do próprio LightGBM. "
        "Positivo empurra para risco, negativo puxa para segurança."
    )
    tabela_exp = explain.explain_one(modelo, X, i, top_n=8)

    fig2 = go.Figure(go.Bar(
        x=tabela_exp["impacto"],
        y=tabela_exp["fator"],
        orientation="h",
        marker_color=[VERMELHO if v > 0 else VERDE for v in tabela_exp["impacto"]],
        hovertemplate="%{y}<br>impacto %{x:+.3f}<extra></extra>",
    ))
    fig2.update_layout(
        xaxis_title="Impacto no escore (log-odds)",
        height=340, margin=dict(t=20, b=40, l=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig2, width='stretch')

    if not aprovada:
        st.markdown("#### Motivos da recusa")
        st.caption(
            "Em crédito, negar sem justificar é problema jurídico antes de ser "
            "problema técnico. Estes são os reason codes que iriam para a carta."
        )
        for m in explain.reason_codes(modelo, X.iloc[[i]], top_n=4)[0]:
            st.markdown(f"- {m['motivo'].capitalize()}")

    with st.expander("Ver todas as features da conta"):
        st.dataframe(
            pd.DataFrame({"feature": X.columns, "valor": X.iloc[i].astype(str).to_numpy()}),
            width='stretch', height=420,
        )
