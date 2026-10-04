# Laboratorio 7 – NLP end-to-end y Embeddings (CC3092)

Pipeline completo: WikiText-103 → tokens → SGNS propio (PyTorch) / Word2Vec (gensim) / GloVe-100 → analogías y similitud → clasificación de AG News.

| Ruta | Contenido |
|---|---|
| `Lab7_NLP_Embeddings.ipynb` | Notebook completo y ejecutado (preprocesamiento, SGNS, iteraciones, analogías, t-SNE, clasificación, comparación y discusión). |
| `informe/informe.tex`, `informe/informe.pdf` | Informe (investigación, resultados y conclusiones). Las tablas `informe/tabla_*.tex` las genera el notebook. |
| `vectors/sgns_best_d100.kv` | Vectores del mejor SGNS propio (100 dim, 100 844 palabras). Cargar con `gensim.models.KeyedVectors.load(...)`. |
| `figures/` | Figuras generadas por el notebook (`curvas_compacto.png` la genera `tools/report_figs.py`). |
| `tools/` | Scripts con los que se construyó el notebook (`parts/p*.py` → `build_nb.py`). |

## Reproducir
```bash
pip install torch gensim nltk datasets scikit-learn matplotlib pandas scipy jupyter
jupyter nbconvert --to notebook --execute --inplace Lab7_NLP_Embeddings.ipynb
```
Los resultados intermedios se guardan en `artifacts/` (no versionado); sin esa carpeta el entrenamiento completo del SGNS tarda ≈ 1 h en una GTX 1660 SUPER.

## Informe
```bash
cd informe && latexmk -pdf informe.tex
```
