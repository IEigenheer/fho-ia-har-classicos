# Filtragem por correlação no reconhecimento de atividades humanas com classificadores clássicos

Trabalho prático da disciplina **Inteligência Artificial I** (FHO, 2026) — eixo
*Classificação e Dados Tabulares*.

**Autores:** Samuel Prado Costa (RA 115116) e Isaac Loureiro Eigenheer (RA 115095)

O projeto avalia se remover atributos redundantes (correlação absoluta de
Pearson ≥ 0,95) altera o desempenho de cinco classificadores clássicos —
Extra Trees, Random Forest, LightGBM, SVM (RBF) e KNN — no reconhecimento de
seis atividades humanas a partir dos sinais inerciais de smartphone da base
[UCI-HAR](https://doi.org/10.24432/C54S4K). A avaliação usa validação cruzada
em 5 dobras **agrupadas por participante** (nenhum voluntário aparece em
treino e teste ao mesmo tempo).

O artigo (template IEEE Conference) está em [`artigo/main.pdf`](artigo/main.pdf).

## Estrutura

```
src/
  dados.py        download e leitura da UCI-HAR (sinais brutos 128 x 9)
  atributos.py    extração de 344 atributos de tempo, frequência e multieixo
  selecao.py      filtro de correlação de Pearson (ajustado só no treino)
  modelos.py      configurações fixas dos 5 classificadores
  experimento.py  validação cruzada GroupKFold (5 dobras) e métricas
  graficos.py     figuras e tabelas do artigo a partir de resultados/
  demo.py         demonstração: treina e classifica janelas de participantes não vistos
resultados/       métricas por dobra, predições, seleção de atributos, versões
artigo/           main.tex, referencias.bib, figuras/, tabelas/, main.pdf
```

## Como executar

Requer Python 3.11+ (testado com 3.13). A base UCI-HAR (~60 MB) é baixada
automaticamente na primeira execução para `dados/brutos/`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/demo.py                 # demonstração rápida (~1 min)
python src/experimento.py          # experimento completo (6 condições x 5 dobras x 5 modelos)
python src/experimento.py --rapido # apenas sem filtro e limiar 0,95
python src/graficos.py             # regenera figuras e tabelas em artigo/
```

Opções da demonstração: `--modelo "SVM (RBF)"`, `--limiar 0` (sem filtro),
`--janelas 20`.

Para compilar o artigo (pdfLaTeX + BibTeX):

```bash
cd artigo
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Pipeline

1. **Dados:** 10 299 janelas de 2,56 s (128 amostras a 50 Hz, 50% de
   sobreposição), 30 participantes, 6 classes (caminhar, subir escadas,
   descer escadas, sentado, em pé, deitado). Canais: aceleração total,
   aceleração do corpo e giroscópio (3 eixos cada).
2. **Atributos (344):** para cada um dos 9 canais e das 3 magnitudes
   vetoriais, 20 estatísticas temporais (média, desvio, percentis, RMS,
   assimetria, curtose, taxa de cruzamento da média, autocorrelação, ...) e 8
   espectrais (energia total e em 0,5–2/2–5/5–10 Hz, frequência de pico,
   centroide, entropia e planicidade espectral); mais pitch/roll médios e
   correlações entre eixos.
3. **Filtro de correlação:** ajustado no treino de cada dobra; em cada par
   com |r| ≥ limiar, descarta o atributo de maior correlação média com os
   demais.
4. **Modelos (hiperparâmetros fixos, semente 42):** Extra Trees (200
   árvores), Random Forest (200 árvores, profundidade 20), LightGBM (200
   iterações, taxa 0,05), SVM RBF (C = 10, gamma = 'scale') e KNN (k = 5),
   os dois últimos com padronização z-score ajustada no treino.
5. **Avaliação:** GroupKFold (5 dobras, participante como grupo); acurácia,
   precisão, revocação e F1 macro por dobra (média ± desvio padrão).

## Resultados (F1 macro, média ± desvio padrão entre as 5 dobras)

| Modelo        | Original (344 atrib.) | Filtrado (≈192 atrib.) | ΔF1 pareado (p.p.) | Dobras com ganho |
|---------------|-----------------------|------------------------|--------------------|------------------|
| Extra Trees   | 92,85 ± 2,96          | 93,88 ± 2,44           | +1,03 ± 0,71       | 5/5              |
| Random Forest | 93,14 ± 2,24          | 93,65 ± 2,71           | +0,52 ± 0,73       | 4/5              |
| LightGBM      | 95,04 ± 2,17          | 95,08 ± 2,12           | +0,04 ± 0,38       | 2/5              |
| SVM (RBF)     | 93,68 ± 2,59          | 93,24 ± 2,76           | −0,44 ± 0,30       | 0/5              |
| KNN (k=5)     | 90,33 ± 1,81          | 89,08 ± 2,06           | −1,24 ± 0,40       | 0/5              |

- O filtro (|r| ≥ 0,95) reduz os atributos em 44,3% e o tempo de treino do
  LightGBM em 43%, sem perda de desempenho.
- Comitês de árvores mantêm o desempenho até |r| ≥ 0,80 (98 atributos);
  SVM e KNN pioram à medida que atributos são removidos, sobretudo nas
  posturas estáticas (sentado, em pé, deitado).

Arquivos completos: `resultados/metricas_dobras.csv`, `resultados/resumo.csv`,
`resultados/predicoes.csv.gz`, `resultados/selecao_0.95.csv`,
`resultados/ambiente.json` (versões e duração: ~11 min em CPU de 16 threads).
