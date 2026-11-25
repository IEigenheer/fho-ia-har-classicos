"""Configurações fixas dos cinco classificadores clássicos.

Os hiperparâmetros são fixados a priori (sem busca), idênticos nas condições
com e sem filtragem. SVM e KNN recebem padronização z-score dos atributos,
ajustada no treino, por dependerem de distâncias; os modelos de árvores usam
os atributos sem escala, pois suas divisões são invariantes a transformações
monótonas.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEMENTE = 42

NOMES_MODELOS = ["Extra Trees", "Random Forest", "LightGBM", "SVM (RBF)", "KNN (k=5)"]


def criar_modelo(nome: str):
    if nome == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=200, criterion="gini", max_features="sqrt",
                                    n_jobs=-1, random_state=SEMENTE)
    if nome == "Random Forest":
        return RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=20,
                                      max_features="sqrt", bootstrap=True,
                                      n_jobs=-1, random_state=SEMENTE)
    if nome == "LightGBM":
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                              random_state=SEMENTE, n_jobs=-1, verbose=-1)
    if nome == "SVM (RBF)":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
    if nome == "KNN (k=5)":
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=5, metric="euclidean"))
    raise ValueError(nome)
