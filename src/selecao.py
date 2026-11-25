"""Filtro de atributos por correlação absoluta de Pearson.

O filtro é ajustado apenas na partição de treino e aplicado ao teste
(transformador compatível com scikit-learn).

Regra: (1) descartam-se colunas de variância nula; (2) calcula-se a matriz
|r| e, para cada atributo, a correlação absoluta média com os demais;
(3) os atributos são percorridos em ordem decrescente dessa média (desempate
pela ordem de extração) e um atributo é descartado se tiver |r| >= limiar com
algum atributo posterior ainda não descartado. Assim, em cada par redundante,
sai o atributo mais correlacionado com o restante do conjunto.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FiltroCorrelacao(TransformerMixin, BaseEstimator):
    def __init__(self, limiar: float | None = 0.95):
        self.limiar = limiar

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_atrib = X.shape[1]
        variavel = X.std(axis=0) > 1e-12
        if self.limiar is None:
            self.mascara_ = variavel
            self.n_atributos_ = int(variavel.sum())
            return self
        idx = np.flatnonzero(variavel)
        C = np.abs(np.corrcoef(X[:, idx], rowvar=False))
        np.fill_diagonal(C, 0.0)
        media = C.sum(axis=1) / (len(idx) - 1)
        ordem = np.argsort(-media, kind="stable")
        removido = np.zeros(len(idx), dtype=bool)
        for pos, i in enumerate(ordem):
            if removido[i]:
                continue
            seguintes = ordem[pos + 1:]
            seguintes = seguintes[~removido[seguintes]]
            if np.any(C[i, seguintes] >= self.limiar):
                removido[i] = True
        mascara = np.zeros(n_atrib, dtype=bool)
        mascara[idx[~removido]] = True
        self.mascara_ = mascara
        self.n_atributos_ = int(mascara.sum())
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mascara_]
