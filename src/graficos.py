"""Gera figuras (PDF) e tabelas (LaTeX) do artigo a partir de resultados/.

Uso:  python src/graficos.py
Saídas: artigo/figuras/*.pdf, artigo/tabelas/*.tex, resultados/resumo.csv,
        resultados/numeros.json (valores citados no texto).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dados import ATIVIDADES, RAIZ  # noqa: E402
from modelos import NOMES_MODELOS  # noqa: E402

RES = RAIZ / "resultados"
FIG = RAIZ / "artigo" / "figuras"
TAB = RAIZ / "artigo" / "tabelas"
PRINCIPAL = "0.95"

AZUL, LARANJA = "#2a78d6", "#eb6834"
CORES_MODELOS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
MARCADORES = ["o", "s", "^", "D", "v"]
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
LARGURA = 3.45  # polegadas (uma coluna IEEE)


def br(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def resumo(m: pd.DataFrame) -> pd.DataFrame:
    met = ["acuracia", "precisao_macro", "revocacao_macro", "f1_macro",
           "tempo_treino_s", "tempo_pred_ms_por_janela", "n_atributos"]
    r = m.groupby(["modelo", "condicao"])[met].agg(["mean", "std"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    return r.reset_index()


def tabela_resultados(r: pd.DataFrame) -> str:
    linhas = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
              r"Modelo & Cond. & Acur. & Prec. & Revoc. & F1 macro\\", r"\midrule"]
    for i, nome in enumerate(NOMES_MODELOS):
        for cond, rot in (("original", "Orig."), (PRINCIPAL, "Filt.")):
            s = r[(r.modelo == nome) & (r.condicao == cond)].iloc[0]
            cel = [f"{br(s[f'{k}_mean'], 1)}$\\pm${br(s[f'{k}_std'], 1)}"
                   for k in ("acuracia", "precisao_macro", "revocacao_macro", "f1_macro")]
            linhas.append(f"{nome if cond == 'original' else ''} & {rot} & " + " & ".join(cel) + r"\\")
        if i < len(NOMES_MODELOS) - 1:
            linhas.append(r"\addlinespace[2pt]")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(linhas) + "\n"


def tabela_deltas(m: pd.DataFrame, r: pd.DataFrame) -> tuple[str, dict]:
    linhas = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
              r"Modelo & $\Delta$F1 (p.p.) & Ganhos & \multicolumn{2}{c}{Treino (s)} & Pred. (ms)\\",
              r"\cmidrule(lr){4-5}", r" & & & Orig. & Filt. & Filt.\\", r"\midrule"]
    info = {}
    for nome in NOMES_MODELOS:
        o = m[(m.modelo == nome) & (m.condicao == "original")].sort_values("dobra")
        f = m[(m.modelo == nome) & (m.condicao == PRINCIPAL)].sort_values("dobra")
        d = f.f1_macro.values - o.f1_macro.values
        ganhos = int(np.sum(d > 0))
        so = r[(r.modelo == nome) & (r.condicao == "original")].iloc[0]
        sf = r[(r.modelo == nome) & (r.condicao == PRINCIPAL)].iloc[0]
        sinal = "+" if d.mean() >= 0 else "$-$"
        linhas.append(f"{nome} & {sinal}{br(abs(d.mean()))}$\\pm${br(d.std(ddof=1))} & {ganhos}/5 & "
                      f"{br(so.tempo_treino_s_mean, 1)} & {br(sf.tempo_treino_s_mean, 1)} & "
                      f"{br(sf.tempo_pred_ms_por_janela_mean, 3)}\\\\")
        info[nome] = {"delta_f1_media": d.mean(), "delta_f1_dp": d.std(ddof=1),
                      "delta_f1_min": d.min(), "delta_f1_max": d.max(), "dobras_com_ganho": ganhos,
                      "treino_orig_s": so.tempo_treino_s_mean, "treino_filt_s": sf.tempo_treino_s_mean,
                      "pred_orig_ms": so.tempo_pred_ms_por_janela_mean,
                      "pred_filt_ms": sf.tempo_pred_ms_por_janela_mean}
    linhas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(linhas) + "\n", info


def f1_por_classe(p: pd.DataFrame) -> tuple[str, dict]:
    classes = list(ATIVIDADES)
    abrev = ["Cam.", "Sub.", "Desc.", "Sent.", "Em pé", "Deit."]
    linhas = [r"\begin{tabular}{@{}l" + "c" * 6 + r"@{}}", r"\toprule",
              "Modelo & " + " & ".join(abrev) + r"\\", r"\midrule"]
    info = {}
    for nome in NOMES_MODELOS:
        s = p[(p.modelo == nome) & (p.condicao == PRINCIPAL)]
        f1 = 100 * f1_score(s.y_real, s.y_pred, labels=classes, average=None)
        info[nome] = dict(zip(abrev, f1.round(2).tolist()))
        linhas.append(f"{nome} & " + " & ".join(br(v, 1) for v in f1) + r"\\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(linhas) + "\n", info


def fig_comparacao(m: pd.DataFrame):
    """Diferença pareada de F1 macro (filtrado - original) em cada fold."""
    fig, ax = plt.subplots(figsize=(LARGURA, 1.75))
    ypos = np.arange(len(NOMES_MODELOS))[::-1]
    for y, nome in zip(ypos, NOMES_MODELOS):
        o = m[(m.modelo == nome) & (m.condicao == "original")].sort_values("dobra")
        f = m[(m.modelo == nome) & (m.condicao == PRINCIPAL)].sort_values("dobra")
        d = f.f1_macro.values - o.f1_macro.values
        jitter = np.linspace(-0.16, 0.16, len(d))
        ax.scatter(d, y + jitter, s=16, facecolor="white", edgecolor=AZUL, lw=1.1, zorder=3,
                   label="Fold" if y == ypos[0] else None)
        ax.plot([d.mean()] * 2, [y - 0.3, y + 0.3], color=LARANJA, lw=2.2, zorder=4,
                label="Média dos 5 folds" if y == ypos[0] else None)
    ax.axvline(0, color=TINTA2, lw=0.8, zorder=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−")))
    ax.set_yticks(ypos, NOMES_MODELOS)
    ax.set_xlabel(r"$\Delta$F1 macro (p.p.): filtrado ($|r|\geq0{,}95$) $-$ original")
    ax.grid(axis="x", color=GRADE, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.5)
    fig.savefig(FIG / "comparacao_f1.pdf")
    plt.close(fig)


def fig_sensibilidade(m: pd.DataFrame):
    conds = ["original", "0.99", "0.95", "0.90", "0.85", "0.80"]
    rot = ["sem\nfiltro", "0,99", "0,95", "0,90", "0,85", "0,80"]
    g = m.groupby(["modelo", "condicao"]).agg(f1=("f1_macro", "mean"), d=("n_atributos", "mean"))
    fig, ax = plt.subplots(figsize=(LARGURA, 2.0))
    x = np.arange(len(conds))
    for nome, cor, mk in zip(NOMES_MODELOS, CORES_MODELOS, MARCADORES):
        v = [g.loc[(nome, c), "f1"] for c in conds]
        ax.plot(x, v, color=cor, lw=1.5, marker=mk, ms=4, label=nome)
    dims = [g.xs(c, level="condicao")["d"].mean() for c in conds]
    ax.set_xticks(x, [f"{a}\n(D={d:.0f})" if i else f"{a}\n(D={d:.0f})" for i, (a, d) in
                      enumerate(zip([r.replace("\n", " ") for r in rot], dims))])
    ax.set_xlabel(r"Limiar de correlação absoluta (D = atributos retidos, média dos folds)")
    ax.set_ylabel("F1 macro (%)")
    ax.grid(axis="y", color=GRADE, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.6, columnspacing=0.8)
    fig.savefig(FIG / "sensibilidade_limiar.pdf")
    plt.close(fig)


def fig_confusao(p: pd.DataFrame, nome: str):
    s = p[(p.modelo == nome) & (p.condicao == PRINCIPAL)]
    classes = list(ATIVIDADES)
    cm = confusion_matrix(s.y_real, s.y_pred, labels=classes)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(LARGURA, 2.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#f7f9fc", "#9ec3ef", "#2a78d6", "#0d3d73"])
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    nomes = [ATIVIDADES[c] for c in classes]
    ax.set_xticks(range(6), nomes, rotation=35, ha="right")
    ax.set_yticks(range(6), nomes)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    for i in range(6):
        for j in range(6):
            if cm[i, j]:
                cor = "white" if cmn[i, j] > 0.55 else TINTA
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.5, color=cor)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(FIG / "confusao_melhor.pdf")
    plt.close(fig)
    return cm


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RES / "metricas_dobras.csv", dtype={"condicao": str})
    p = pd.read_csv(RES / "predicoes.csv.gz", dtype={"condicao": str})
    r = resumo(m)
    r.to_csv(RES / "resumo.csv", index=False, float_format="%.4f")

    (TAB / "resultados.tex").write_text(tabela_resultados(r), encoding="utf-8")
    tex, deltas = tabela_deltas(m, r)
    (TAB / "deltas.tex").write_text(tex, encoding="utf-8")
    tex, classes = f1_por_classe(p)
    (TAB / "f1_classes.tex").write_text(tex, encoding="utf-8")

    filt = r[r.condicao == PRINCIPAL].sort_values("f1_macro_mean", ascending=False)
    melhor = filt.iloc[0].modelo
    fig_comparacao(m)
    if m.condicao.nunique() > 2:
        fig_sensibilidade(m)
    cm = fig_confusao(p, melhor)

    sel = pd.read_csv(RES / f"selecao_{PRINCIPAL}.csv")
    numeros = {
        "melhor_modelo_filtrado": melhor,
        "resumo": r.round(4).to_dict(orient="records"),
        "deltas": deltas, "f1_por_classe_filtrado": classes,
        "confusao_melhor": cm.tolist(),
        "atributos_retidos_5_dobras": int((sel.dobras_retido == 5).sum()),
        "atributos_retidos_alguma_dobra": int((sel.dobras_retido > 0).sum()),
        "f1_por_limiar": m.groupby(["modelo", "condicao"]).f1_macro.mean().round(2).unstack().to_dict(),
        "atributos_por_limiar": m.groupby("condicao").n_atributos.agg(["mean", "min", "max"]).to_dict(),
    }
    (RES / "numeros.json").write_text(json.dumps(numeros, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    print(f"Figuras em {FIG}, tabelas em {TAB}. Melhor modelo filtrado: {melhor}")


if __name__ == "__main__":
    main()
