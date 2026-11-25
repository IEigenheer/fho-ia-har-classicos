"""Download e leitura da base UCI-HAR (sinais inerciais brutos).

A base é distribuída com uma divisão oficial treino/teste. Este projeto reúne
as duas partes (10 299 janelas, 30 participantes) e realiza sua própria
validação cruzada agrupada por participante (ver experimento.py).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / "dados" / "brutos"
PASTA_UCI = PASTA_DADOS / "UCI HAR Dataset"
URL_UCI = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

# Canais usados: aceleração total (g), aceleração do corpo (g, sem gravidade)
# e velocidade angular do giroscópio (rad/s), cada um em três eixos.
CANAIS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

ATIVIDADES = {
    1: "Caminhar",
    2: "Subir escadas",
    3: "Descer escadas",
    4: "Sentado",
    5: "Em pé",
    6: "Deitado",
}

FREQ_AMOSTRAGEM = 50.0  # Hz
AMOSTRAS_JANELA = 128   # 2,56 s com 50% de sobreposição


def baixar_uci_har(forcar: bool = False) -> Path:
    """Baixa e extrai a UCI-HAR em dados/brutos, caso ainda não exista."""
    if PASTA_UCI.exists() and not forcar:
        return PASTA_UCI
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    print(f"Baixando UCI-HAR de {URL_UCI} ...")
    with urllib.request.urlopen(URL_UCI) as resp:
        externo = zipfile.ZipFile(io.BytesIO(resp.read()))
    # O arquivo do repositório UCI contém um segundo .zip com os dados.
    interno_nome = next(n for n in externo.namelist() if n.endswith("UCI HAR Dataset.zip"))
    with zipfile.ZipFile(io.BytesIO(externo.read(interno_nome))) as interno:
        membros = [m for m in interno.namelist() if not m.startswith("__MACOSX")]
        interno.extractall(PASTA_DADOS, members=membros)
    print(f"Base extraída em {PASTA_UCI}")
    return PASTA_UCI


def _ler_particao(particao: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pasta = PASTA_UCI / particao
    sinais = []
    for canal in CANAIS:
        arq = pasta / "Inertial Signals" / f"{canal}_{particao}.txt"
        sinais.append(np.loadtxt(arq, dtype=np.float64))
    X = np.stack(sinais, axis=-1)  # (janelas, 128, 9)
    y = np.loadtxt(pasta / f"y_{particao}.txt", dtype=int)
    grupos = np.loadtxt(pasta / f"subject_{particao}.txt", dtype=int)
    return X, y, grupos


def carregar_sinais() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (X, y, participante) com as partições oficiais concatenadas.

    X tem forma (10299, 128, 9), na ordem de CANAIS.
    """
    baixar_uci_har()
    Xtr, ytr, gtr = _ler_particao("train")
    Xte, yte, gte = _ler_particao("test")
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    g = np.concatenate([gtr, gte])
    assert X.shape[1:] == (AMOSTRAS_JANELA, len(CANAIS))
    return X, y, g


if __name__ == "__main__":
    X, y, g = carregar_sinais()
    print("Sinais:", X.shape, "| participantes:", len(np.unique(g)))
    for k, v in ATIVIDADES.items():
        print(f"  {k} {v:15s} {np.sum(y == k):5d} janelas")
