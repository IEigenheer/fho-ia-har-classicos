"""Experimento principal: efeito da filtragem por correlação em 5 classificadores.

Protocolo
  * UCI-HAR completa (10 299 janelas, 30 participantes, 6 classes);
  * validação cruzada GroupKFold com 5 dobras, participante como grupo
    (nenhum participante aparece em treino e teste na mesma dobra);
  * em cada dobra: filtro de correlação ajustado no treino -> modelo -> teste;
  * condições: sem filtro (original) e limiares |r| >= 0,99; 0,95; 0,90; 0,85; 0,80.
    O limiar principal (0,95) foi fixado a priori; os demais formam uma análise
    de sensibilidade e não são usados para escolher configuração.

Saídas em resultados/:
  dobras.csv            participantes de teste por dobra
  metricas_dobras.csv   métricas por modelo, condição e dobra
  predicoes.csv.gz      predições de todas as janelas (condições original e 0,95)
  selecao_0.95.csv      quantas dobras retiveram cada atributo no limiar 0,95
  ambiente.json         versões das bibliotecas e duração

Uso:  python src/experimento.py            (todas as condições)
      python src/experimento.py --rapido   (apenas original e 0,95)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atributos import extrair  # noqa: E402
from dados import RAIZ, carregar_sinais  # noqa: E402
from modelos import NOMES_MODELOS, SEMENTE, criar_modelo  # noqa: E402
from selecao import FiltroCorrelacao  # noqa: E402

PASTA_RES = RAIZ / "resultados"
CACHE_ATRIB = RAIZ / "dados" / "atributos_uci_har.npz"
LIMIARES = [None, 0.99, 0.95, 0.90, 0.85, 0.80]
LIMIAR_PRINCIPAL = 0.95
N_DOBRAS = 5


def rotulo(limiar):
    return "original" if limiar is None else f"{limiar:.2f}"


def obter_atributos():
    if CACHE_ATRIB.exists():
        d = np.load(CACHE_ATRIB, allow_pickle=False)
        return d["F"], list(d["nomes"]), d["y"], d["g"]
    X, y, g = carregar_sinais()
    t0 = time.perf_counter()
    F, nomes = extrair(X)
    print(f"Extração: {F.shape[1]} atributos em {time.perf_counter() - t0:.1f} s")
    CACHE_ATRIB.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_ATRIB, F=F, nomes=np.array(nomes), y=y, g=g)
    return F, nomes, y, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="apenas original e 0,95")
    args = ap.parse_args()
    limiares = [None, LIMIAR_PRINCIPAL] if args.rapido else LIMIARES

    PASTA_RES.mkdir(exist_ok=True)
    inicio = time.perf_counter()
    F, nomes, y, g = obter_atributos()
    print(f"Matriz: {F.shape[0]} janelas x {F.shape[1]} atributos; "
          f"{len(np.unique(g))} participantes")

    dobras = list(GroupKFold(n_splits=N_DOBRAS).split(F, y, g))
    pd.DataFrame([{"dobra": k + 1, "participantes_teste": " ".join(map(str, sorted(np.unique(g[te])))),
                   "janelas_treino": len(tr), "janelas_teste": len(te)}
                  for k, (tr, te) in enumerate(dobras)]).to_csv(PASTA_RES / "dobras.csv", index=False)

    linhas, predicoes = [], []
    retencao = np.zeros(len(nomes), dtype=int)
    for k, (tr, te) in enumerate(dobras, start=1):
        for limiar in limiares:
            t0 = time.perf_counter()
            filtro = FiltroCorrelacao(limiar).fit(F[tr])
            t_filtro = time.perf_counter() - t0
            Ftr, Fte = filtro.transform(F[tr]), filtro.transform(F[te])
            if limiar == LIMIAR_PRINCIPAL:
                retencao += filtro.mascara_
            for nome in NOMES_MODELOS:
                modelo = criar_modelo(nome)
                t0 = time.perf_counter()
                modelo.fit(Ftr, y[tr])
                t_treino = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred = modelo.predict(Fte)
                t_pred = time.perf_counter() - t0
                linhas.append({
                    "modelo": nome, "condicao": rotulo(limiar), "dobra": k,
                    "n_atributos": filtro.n_atributos_,
                    "acuracia": 100 * accuracy_score(y[te], pred),
                    "precisao_macro": 100 * precision_score(y[te], pred, average="macro", zero_division=0),
                    "revocacao_macro": 100 * recall_score(y[te], pred, average="macro", zero_division=0),
                    "f1_macro": 100 * f1_score(y[te], pred, average="macro"),
                    "tempo_filtro_s": t_filtro, "tempo_treino_s": t_treino,
                    "tempo_pred_ms_por_janela": 1000 * t_pred / len(te),
                })
                if limiar in (None, LIMIAR_PRINCIPAL):
                    predicoes.append(pd.DataFrame({
                        "modelo": nome, "condicao": rotulo(limiar), "dobra": k,
                        "indice": te, "participante": g[te], "y_real": y[te], "y_pred": pred}))
                r = linhas[-1]
                print(f"dobra {k} | {rotulo(limiar):8s} | D={r['n_atributos']:3d} | {nome:13s} | "
                      f"F1={r['f1_macro']:6.2f} | treino {t_treino:5.1f}s")

    pd.DataFrame(linhas).to_csv(PASTA_RES / "metricas_dobras.csv", index=False, float_format="%.4f")
    pd.concat(predicoes).to_csv(PASTA_RES / "predicoes.csv.gz", index=False)
    pd.DataFrame({"atributo": nomes, "dobras_retido": retencao}).to_csv(
        PASTA_RES / f"selecao_{LIMIAR_PRINCIPAL:.2f}.csv", index=False)

    import lightgbm, scipy, sklearn  # noqa: E401
    ambiente = {
        "python": platform.python_version(), "sistema": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
        "semente": SEMENTE, "dobras": N_DOBRAS, "limiares": [rotulo(l) for l in limiares],
        "duracao_total_s": round(time.perf_counter() - inicio, 1),
    }
    (PASTA_RES / "ambiente.json").write_text(json.dumps(ambiente, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"Concluído em {ambiente['duracao_total_s']} s. Resultados em {PASTA_RES}")


if __name__ == "__main__":
    main()
