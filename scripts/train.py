"""Pipeline completo: dado bruto -> modelo calibrado -> política de corte.

Uso:
    python -m scripts.train

Gera em reports/ as tabelas e figuras citadas no README, e em models/ o
artefato servido pelo app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, decision, explain, features, model, monitoring  # noqa: E402

# O console do Windows abre em cp1252 e engasga com acento.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

AZUL, VERMELHO, VERDE, CINZA = "#1f4e79", "#c0392b", "#27ae60", "#7f8c8d"
REFERENCIA = "Threshold 0,50 (default)"


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> None:
    print("\n=== 1. Dados ===")
    df = data.load()
    log(f"{len(df):,} contas carregadas".replace(",", "."))

    X, y, aux = features.build(df)
    log(f"{X.shape[1]} features construídas | inadimplência global {y.mean():.2%}")
    log("fora das features por serem atributos protegidos: sexo, estado_civil")

    print("\n=== 2. Partições ===")
    (X_tr, y_tr, _), (X_es, y_es, _), (X_ca, y_ca, aux_ca), (X_te, y_te, aux_te) = model.split(
        X, y, aux
    )
    log(f"treino {len(X_tr):,} | early-stop {len(X_es):,} | "
        f"calibração {len(X_ca):,} | teste {len(X_te):,}".replace(",", "."))

    print("\n=== 3. Treino ===")
    risk_model, info = model.train(X_tr, y_tr, X_es, y_es, X_ca, y_ca)
    log(f"melhor iteração: {info['melhor_iteracao']}")

    print("\n=== 4. Seleção do calibrador (Brier em conjunto de seleção) ===")
    sel = info["selecao_calibrador"]
    for _, r in sel.iterrows():
        marca = "  <-- escolhido" if r["calibrador"] == risk_model.calibrator_name else ""
        log(f"{r['calibrador']:<10} Brier {r['brier_selecao']:.5f} | "
            f"AUC {r['auc_selecao']:.4f} | PD média {r['pd_media']:.4f} "
            f"vs observada {r['taxa_observada']:.4f}{marca}")
    sel.to_csv(config.REPORTS / "selecao_calibrador.csv", index=False)

    p_raw_te = risk_model.raw_score(X_te)
    p_cal_te = risk_model.predict_pd(X_te)
    p_cal_ca = risk_model.predict_pd(X_ca)

    m_raw = model.evaluate(y_te.to_numpy(), p_raw_te)
    m_cal = model.evaluate(y_te.to_numpy(), p_cal_te)

    print("\n=== 5. Desempenho no teste ===")
    log(f"escore bruto   AUC {m_raw['auc']:.4f} | KS {m_raw['ks']:.4f} | Brier {m_raw['brier']:.5f}")
    log(f"servido ({risk_model.calibrator_name})  AUC {m_cal['auc']:.4f} | "
        f"KS {m_cal['ks']:.4f} | Brier {m_cal['brier']:.5f}")
    log(f"PD média prevista {m_cal['pd_media_prevista']:.4f} vs "
        f"observada {m_cal['taxa_observada']:.4f} "
        f"(desvio {m_cal['pd_media_prevista'] - m_cal['taxa_observada']:+.4f})")

    tab_cal = model.calibration_table(y_te.to_numpy(), p_cal_te)
    tab_cal.to_csv(config.REPORTS / "tabela_calibracao.csv", index=False)
    log(f"maior desvio por decil: {tab_cal['desvio'].abs().max():.4f}")

    print("\n=== 6. Camada de decisão ===")
    exp_te = aux_te["exposicao"].to_numpy()
    exp_ca = aux_ca["exposicao"].to_numpy()

    t_breakeven = decision.breakeven_pd()
    t_f1 = decision.max_f1_threshold(p_cal_ca, y_ca.to_numpy())
    t_emp = decision.empirical_optimum(p_cal_ca, exp_ca, y_ca.to_numpy())

    log(f"break-even analítico  pd* = m/(m+lgd) = {config.ANNUAL_MARGIN_RATE}/"
        f"({config.ANNUAL_MARGIN_RATE}+{config.LGD}) = {t_breakeven:.4f}")
    log(f"ótimo empírico (calibração)            = {t_emp:.4f}")
    log(f"máxima F1 (calibração)                 = {t_f1:.4f}")

    politicas = {
        "Aprovar todos": 1.01,
        REFERENCIA: 0.50,
        "Máxima F1": t_f1,
        "Break-even econômico": t_breakeven,
        "Ótimo empírico (calib)": t_emp,
    }
    comp = decision.compare_policies(
        p_cal_te, exp_te, y_te.to_numpy(), politicas, referencia=REFERENCIA
    )
    comp.to_csv(config.REPORTS / "comparacao_politicas.csv")

    print()
    vis = comp[
        ["threshold", "taxa_aprovacao", "inadimplencia_da_carteira", "lucro_realizado", "ganho_pct"]
    ].copy()
    vis["lucro_realizado"] = vis["lucro_realizado"].map(decision.format_currency)
    vis["taxa_aprovacao"] = vis["taxa_aprovacao"].map(lambda v: f"{v:.1%}")
    vis["inadimplencia_da_carteira"] = vis["inadimplencia_da_carteira"].map(lambda v: f"{v:.2%}")
    vis["ganho_pct"] = vis["ganho_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1%}")
    vis["threshold"] = vis["threshold"].map(lambda v: f"{v:.3f}")
    vis.columns = ["corte", "aprovação", "inadimpl. carteira", "lucro", "vs 0,50"]
    print(vis.to_string())

    curva = decision.profit_curve(p_cal_te, exp_te, y_te.to_numpy())
    curva.to_csv(config.REPORTS / "curva_lucro.csv", index=False)

    # Largura do platô: faixa de thresholds dentro de 1% do lucro máximo.
    topo = curva["lucro_realizado"].max()
    plato = curva[curva["lucro_realizado"] >= topo * 0.99]["threshold"]
    log(f"platô a 1% do ótimo: threshold de {plato.min():.3f} a {plato.max():.3f}")

    print("\n=== 7. Fairness no corte escolhido ===")
    fair = monitoring.fairness_report(
        aux_te.reset_index(drop=True),
        p_cal_te,
        y_te.to_numpy(),
        t_breakeven,
        config.PROTECTED_ATTRIBUTES,
    )
    fair.to_csv(config.REPORTS / "fairness.csv", index=False)
    if not fair.empty:
        pior = fair.loc[fair["indice_paridade"].idxmin()]
        log(f"menor índice de paridade: {pior['atributo']}={pior['grupo']} "
            f"({pior['indice_paridade']:.3f})")
        log(f"    aprovação {pior['taxa_aprovacao']:.1%} | "
            f"inadimplência observada do grupo {pior['inadimplencia_observada']:.1%} "
            f"(base {y_te.mean():.1%})")

    print("\n=== 8. Estabilidade ===")
    psi_te = monitoring.psi(p_cal_ca, p_cal_te)
    log(f"PSI do escore, calibração -> teste: {psi_te:.4f} ({monitoring.classify_psi(psi_te)})")

    X_drift = monitoring.simulate_drift(X_te, fracao=0.30)
    p_drift = risk_model.predict_pd(X_drift)
    psi_drift = monitoring.psi(p_cal_te, p_drift)
    log(f"PSI sob safra deteriorada (30% da base): {psi_drift:.4f} "
        f"({monitoring.classify_psi(psi_drift)})")

    csi = monitoring.csi_table(X_te, X_drift)
    csi.to_csv(config.REPORTS / "csi.csv", index=False)
    log("features que mais se deslocaram: " + ", ".join(
        f"{r.feature} ({r.csi:.2f})" for r in csi.head(3).itertuples()
    ))

    # O que o drift faria com a carteira se ninguém recalibrasse o corte.
    res_drift = decision.apply_policy(p_drift, exp_te, y_te.to_numpy(), t_breakeven)
    res_base = comp.loc["Break-even econômico"]
    queda = res_base["taxa_aprovacao"] - res_drift["taxa_aprovacao"]
    log(f"aprovação cairia de {res_base['taxa_aprovacao']:.1%} para "
        f"{res_drift['taxa_aprovacao']:.1%} ({queda:.1%} p.p.) sem ninguém tocar no threshold")
    if monitoring.classify_psi(psi_drift) == "estavel" and abs(queda) > 0.03:
        log("ATENÇÃO: o PSI do escore ficou na faixa estável enquanto a aprovação")
        log("caiu vários pontos. PSI agregado é pouco sensível — é o CSI por")
        log("feature que localiza o deslocamento. Monitorar só o escore não basta.")

    print("\n=== 9. Explicabilidade ===")
    imp = explain.global_importance(risk_model, X_te)
    imp.to_csv(config.REPORTS / "importancia_global.csv", index=False)
    log("top 5 fatores: " + ", ".join(imp["feature"].head(5)))

    recusados = np.where(p_cal_te >= t_breakeven)[0]
    if len(recusados):
        exemplo = int(recusados[0])
        motivos = explain.reason_codes(risk_model, X_te.iloc[[exemplo]])[0]
        log(f"exemplo de recusa (PD {p_cal_te[exemplo]:.1%}):")
        for m in motivos:
            log(f"    - {m['motivo']} ({m['impacto_log_odds']:+.3f})")

    print("\n=== 10. Figuras ===")
    _figuras(y_te.to_numpy(), p_raw_te, p_cal_te, tab_cal, curva, politicas, imp)
    log(f"salvas em {config.FIGURES}")

    print("\n=== 11. Artefatos ===")
    joblib.dump(risk_model, config.MODELS / "modelo_calibrado.joblib")
    X_te.assign(**{config.TARGET: y_te.to_numpy(), "pd": p_cal_te, "exposicao": exp_te}).to_parquet(
        config.DATA_PROCESSED / "teste_scorado.parquet"
    )
    log("modelo em models/modelo_calibrado.joblib")

    linha_be = comp.loc["Break-even econômico"]
    linha_ref = comp.loc[REFERENCIA]
    linha_todos = comp.loc["Aprovar todos"]

    metricas = {
        "modelo": {
            "calibrador_servido": risk_model.calibrator_name,
            "selecao_calibrador": sel.to_dict("records"),
            "teste_bruto": m_raw,
            "teste_servido": m_cal,
            "max_desvio_decil": float(tab_cal["desvio"].abs().max()),
        },
        "premissas": {
            "margem_anual": config.ANNUAL_MARGIN_RATE,
            "lgd": config.LGD,
            "moeda": config.CURRENCY,
            "brl_por_unidade": config.BRL_PER_UNIT,
            "decisao": "gestao de conta (manter ou cortar linha rotativa)",
        },
        "thresholds": {
            "break_even_analitico": t_breakeven,
            "otimo_empirico_calib": t_emp,
            "maxima_f1_calib": t_f1,
            "plato_1pct": [float(plato.min()), float(plato.max())],
        },
        "resultado": {
            "lucro_sem_politica": float(linha_todos["lucro_realizado"]),
            "lucro_threshold_050": float(linha_ref["lucro_realizado"]),
            "lucro_break_even": float(linha_be["lucro_realizado"]),
            "ganho_vs_050_absoluto": float(linha_be["ganho_vs_referencia"]),
            "ganho_vs_050_pct": float(linha_be["ganho_pct"]),
            "taxa_aprovacao": float(linha_be["taxa_aprovacao"]),
            "inadimplencia_da_carteira": float(linha_be["inadimplencia_da_carteira"]),
        },
        "estabilidade": {"psi_calib_teste": psi_te, "psi_drift_simulado": psi_drift},
    }
    (config.REPORTS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ganho = float(linha_be["ganho_vs_referencia"])
    print("\n" + "=" * 74)
    print(f"  Sem política de crédito (aprovar todos): {decision.format_currency(linha_todos['lucro_realizado'])}")
    print(f"  Threshold 0,50, o corte que vem de graça: {decision.format_currency(linha_ref['lucro_realizado'])}")
    print(f"  Corte no break-even econômico ({t_breakeven:.1%}):   {decision.format_currency(linha_be['lucro_realizado'])}")
    print()
    print(f"  Ganho sobre o corte padrão: {decision.format_currency(ganho)} "
          f"({decision.format_currency(ganho, brl=True)}) = {linha_be['ganho_pct']:+.1%}")
    print(f"  Mesmo modelo, mesma AUC. Só o ponto de corte mudou.")
    print("=" * 74 + "\n")


def _figuras(y_te, p_raw, p_cal, tab_cal, curva, politicas, imp) -> None:
    # -- Calibração ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    t = model.calibration_table(y_te, p_cal)
    ax.plot(t["pd_prevista"], t["taxa_observada"], "o-", color=AZUL, lw=1.6, ms=4,
            label="modelo servido")
    ax.plot([0, 1], [0, 1], "--", color=VERMELHO, lw=1, label="calibração perfeita")
    lim = max(tab_cal["pd_prevista"].max(), tab_cal["taxa_observada"].max()) * 1.1
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("PD prevista"); ax.set_ylabel("Inadimplência observada")
    ax.set_title("Calibração por decil de risco")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "calibracao.png"); plt.close(fig)

    # -- Curva de lucro --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(curva["threshold"], curva["lucro_realizado"], color=AZUL, lw=1.8)
    ax.axhline(0, color=CINZA, lw=0.8)
    idx = curva["lucro_realizado"].idxmax()
    ax.axvline(politicas["Break-even econômico"], color=VERMELHO, ls="--", lw=1.3,
               label=f"break-even analítico ({politicas['Break-even econômico']:.3f})")
    ax.axvline(curva.loc[idx, "threshold"], color=VERDE, ls=":", lw=1.5,
               label=f"máximo observado ({curva.loc[idx, 'threshold']:.3f})")
    ax.axvline(0.5, color=CINZA, ls="-.", lw=1.1, label="threshold 0,50 (default)")
    ax.set_xlabel("Threshold de PD (aprova abaixo)")
    ax.set_ylabel(f"Lucro da carteira ({config.CURRENCY})")
    ax.set_title("Resultado financeiro por ponto de corte")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    fig.tight_layout(); fig.savefig(config.FIGURES / "curva_lucro.png"); plt.close(fig)

    # -- Trade-off aprovação x inadimplência -----------------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(curva["taxa_aprovacao"], curva["inadimplencia_da_carteira"], color=AZUL, lw=1.8)
    be = curva.iloc[(curva["threshold"] - politicas["Break-even econômico"]).abs().argmin()]
    ax.scatter([be["taxa_aprovacao"]], [be["inadimplencia_da_carteira"]],
               color=VERMELHO, zorder=5, s=45, label="política escolhida")
    ax.set_xlabel("Taxa de aprovação"); ax.set_ylabel("Inadimplência da carteira aprovada")
    ax.set_title("Fronteira de aprovação x risco")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "fronteira.png"); plt.close(fig)

    # -- Importância -----------------------------------------------------
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.barh(top["feature"], top["impacto_medio"], color=AZUL)
    ax.set_xlabel("|SHAP| médio (log-odds)")
    ax.set_title("Fatores que mais movem o escore")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(config.FIGURES / "importancia.png"); plt.close(fig)


if __name__ == "__main__":
    main()
