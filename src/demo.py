"""Demonstração para a apresentação: treina o pipeline e classifica janelas
de participantes que o modelo nunca viu.

Uso:  python src/demo.py [--modelo "Extra Trees"] [--limiar 0.95] [--janelas 15]

Fluxo: separa 6 participantes para teste (dobra 1 do GroupKFold), extrai os
atributos, ajusta o filtro de correlação e o modelo nos outros 24
participantes e mostra, para janelas sorteadas dos participantes de teste,
a atividade real, a predita e o tempo de inferência por janela.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atributos import extrair  # noqa: E402
from dados import ATIVIDADES, carregar_sinais  # noqa: E402
from modelos import NOMES_MODELOS, criar_modelo  # noqa: E402
from selecao import FiltroCorrelacao  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Extra Trees", choices=NOMES_MODELOS)
    ap.add_argument("--limiar", type=float, default=0.95, help="use 0 para desativar o filtro")
    ap.add_argument("--janelas", type=int, default=15)
    ap.add_argument("--semente", type=int, default=7)
    args = ap.parse_args()
    limiar = args.limiar if args.limiar > 0 else None

    print("1) Carregando sinais da UCI-HAR ...")
    X, y, g = carregar_sinais()
    tr, te = next(GroupKFold(n_splits=5).split(X, y, g))
    print(f"   treino: {len(np.unique(g[tr]))} participantes ({len(tr)} janelas) | "
          f"teste: participantes {sorted(np.unique(g[te]).tolist())} ({len(te)} janelas)")

    print("2) Extraindo atributos de tempo e frequência ...")
    t0 = time.perf_counter()
    F, nomes = extrair(X)
    print(f"   {F.shape[1]} atributos por janela ({time.perf_counter() - t0:.1f} s)")

    print(f"3) Treinando filtro (limiar={limiar}) + {args.modelo} ...")
    pipe = make_pipeline(FiltroCorrelacao(limiar), criar_modelo(args.modelo))
    t0 = time.perf_counter()
    pipe.fit(F[tr], y[tr])
    print(f"   atributos retidos: {pipe[0].n_atributos_} | treino em {time.perf_counter() - t0:.1f} s")

    print("4) Classificando janelas de participantes não vistos:\n")
    rng = np.random.default_rng(args.semente)
    amostra = rng.choice(te, size=args.janelas, replace=False)
    print(f"   {'part.':>5}  {'atividade real':<15} {'predita':<15} {'ms':>6}")
    acertos = 0
    for i in amostra:
        t0 = time.perf_counter()
        janela = extrair(X[i:i + 1])[0]
        pred = pipe.predict(janela)[0]
        ms = 1000 * (time.perf_counter() - t0)
        ok = pred == y[i]
        acertos += ok
        print(f"   {g[i]:>5}  {ATIVIDADES[y[i]]:<15} {ATIVIDADES[pred]:<15} {ms:6.1f}  {'ok' if ok else 'ERRO'}")
    print(f"\n   acertos na amostra: {acertos}/{args.janelas} "
          "(tempo inclui extração de atributos da janela)")

    pred = pipe.predict(F[te])
    print(f"\n5) Todas as {len(te)} janelas de teste: F1 macro = "
          f"{100 * f1_score(y[te], pred, average='macro'):.2f}%\n")
    print(classification_report(y[te], pred, target_names=list(ATIVIDADES.values()), digits=3))


if __name__ == "__main__":
    main()
