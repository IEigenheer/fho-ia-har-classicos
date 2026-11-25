"""Extração de atributos no domínio do tempo e da frequência.

Cada janela (128 amostras x 9 canais) é transformada em um vetor de atributos:
  * 12 sinais = 9 canais + 3 magnitudes vetoriais (SMV) de total_acc, body_acc
    e body_gyro;
  * 20 atributos temporais + 8 espectrais por sinal (12 x 28 = 336);
  * 8 atributos multieixo: pitch e roll médios (a partir de total_acc) e as
    correlações entre eixos de body_acc e body_gyro.
Total: 344 atributos.

A extração é determinística e opera janela a janela, sem parâmetros estimados
nos dados; por isso pode ser feita uma única vez antes da validação cruzada
sem vazamento de informação entre treino e teste.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from dados import AMOSTRAS_JANELA, CANAIS, FREQ_AMOSTRAGEM

EPS = 1e-12

ATRIB_TEMPO = [
    "media", "desvio", "variancia", "mediana", "minimo", "maximo", "pico_a_pico",
    "rms", "energia", "p25", "p75", "iqr", "desvio_abs_medio", "media_abs",
    "assimetria", "curtose", "taxa_cruz_media", "autocorr_l1", "autocorr_l2",
    "fator_crista",
]
ATRIB_FREQ = [
    "energia_espectral", "energia_0.5_2Hz", "energia_2_5Hz", "energia_5_10Hz",
    "freq_pico", "centroide_espectral", "entropia_espectral", "planicidade_espectral",
]
BANDAS = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]


def _autocorr(x: np.ndarray, lag: int) -> np.ndarray:
    xc = x - x.mean(axis=1, keepdims=True)
    num = np.sum(xc[:, :-lag] * xc[:, lag:], axis=1)
    den = np.sum(xc * xc, axis=1)
    return np.where(den > EPS, num / np.maximum(den, EPS), 0.0)


def atributos_tempo(x: np.ndarray) -> np.ndarray:
    """x: (janelas, amostras). Retorna (janelas, 20)."""
    media = x.mean(axis=1)
    desvio = x.std(axis=1)
    p25, mediana, p75 = np.percentile(x, [25, 50, 75], axis=1)
    minimo, maximo = x.min(axis=1), x.max(axis=1)
    energia = np.mean(x ** 2, axis=1)
    rms = np.sqrt(energia)
    xc = x - media[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        assim = np.nan_to_num(stats.skew(x, axis=1))
        curt = np.nan_to_num(stats.kurtosis(x, axis=1))
    sinal = np.sign(xc)
    cruz = np.mean(np.abs(np.diff(sinal, axis=1)) > 0, axis=1)
    crista = np.max(np.abs(x), axis=1) / np.maximum(rms, EPS)
    return np.column_stack([
        media, desvio, desvio ** 2, mediana, minimo, maximo, maximo - minimo,
        rms, energia, p25, p75, p75 - p25, np.mean(np.abs(xc), axis=1),
        np.mean(np.abs(x), axis=1), assim, curt, cruz,
        _autocorr(x, 1), _autocorr(x, 2), crista,
    ])


_FREQS = np.fft.rfftfreq(AMOSTRAS_JANELA, d=1.0 / FREQ_AMOSTRAGEM)


def atributos_frequencia(x: np.ndarray) -> np.ndarray:
    """Espectro de potência da janela sem a componente média. Retorna (janelas, 8)."""
    xc = x - x.mean(axis=1, keepdims=True)
    potencia = np.abs(np.fft.rfft(xc, axis=1)) ** 2 / x.shape[1]
    total = potencia.sum(axis=1)
    bandas = [potencia[:, (_FREQS >= a) & (_FREQS < b)].sum(axis=1) for a, b in BANDAS]
    pico = _FREQS[np.argmax(potencia[:, 1:], axis=1) + 1]
    total_seguro = np.maximum(total, EPS)
    centroide = (potencia * _FREQS).sum(axis=1) / total_seguro
    p = potencia / total_seguro[:, None]
    entropia = -np.sum(p * np.log2(p + EPS), axis=1) / np.log2(p.shape[1])
    pot = potencia[:, 1:] + EPS
    planicidade = np.exp(np.mean(np.log(pot), axis=1)) / np.mean(pot, axis=1)
    return np.column_stack([total, *bandas, pico, centroide, entropia, planicidade])


def _correlacao_eixos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(ac ** 2, axis=1) * np.sum(bc ** 2, axis=1))
    return np.where(den > EPS, np.sum(ac * bc, axis=1) / np.maximum(den, EPS), 0.0)


def extrair(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """X: (janelas, 128, 9) na ordem de CANAIS. Retorna (matriz, nomes)."""
    idx = {c: i for i, c in enumerate(CANAIS)}
    sinais: dict[str, np.ndarray] = {c: X[:, :, idx[c]] for c in CANAIS}
    for grupo in ("total_acc", "body_acc", "body_gyro"):
        eixos = [sinais[f"{grupo}_{e}"] for e in "xyz"]
        sinais[f"{grupo}_mag"] = np.sqrt(sum(e ** 2 for e in eixos))

    blocos, nomes = [], []
    for nome, s in sinais.items():
        blocos += [atributos_tempo(s), atributos_frequencia(s)]
        nomes += [f"{nome}__{a}" for a in ATRIB_TEMPO + ATRIB_FREQ]

    ax, ay, az = (sinais[f"total_acc_{e}"] for e in "xyz")
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))).mean(axis=1)
    roll = np.degrees(np.arctan2(ay, az)).mean(axis=1)
    multi = [pitch, roll]
    nomes += ["multi__pitch_medio", "multi__roll_medio"]
    for grupo in ("body_acc", "body_gyro"):
        for e1, e2 in (("x", "y"), ("x", "z"), ("y", "z")):
            multi.append(_correlacao_eixos(sinais[f"{grupo}_{e1}"], sinais[f"{grupo}_{e2}"]))
            nomes.append(f"multi__{grupo}_corr_{e1}{e2}")
    blocos.append(np.column_stack(multi))

    F = np.column_stack(blocos).astype(np.float64)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, nomes
