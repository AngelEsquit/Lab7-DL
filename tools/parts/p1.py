CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
# Laboratorio 7 — NLP end-to-end y Embeddings
**CC3092 · Deep Learning y Sistemas Inteligentes**

Pipeline completo: texto crudo → tokens → IDs → embeddings (SGNS propio en PyTorch, Word2Vec de gensim, GloVe preentrenado) → evaluación (analogías, similitud) → clasificación de AG News.

**Convenciones del notebook**
- Todo lo costoso se guarda en `artifacts/` (caché). Si el archivo existe, se carga en vez de recalcular.
- Para la **selección de configuraciones** solo se usan WordSim-353 y `questions-words.txt`. **SimLex-999 se reserva** para la evaluación final (sección 6/7).
- Subconjunto de WikiText-103 usado para entrenar: primeras ~30 M de tokens (el enunciado exige ≥ 20 M), el mismo para SGNS y gensim.
""")

code(r"""
import os, re, json, time, math, html, pickle, random, collections, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
VEC = Path("vectors"); VEC.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
SUBSET_TOKENS = 30_000_000          # tamaño objetivo del subconjunto de entrenamiento
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu")
print("torch", torch.__version__)
""")

md(r"""
## 2. Exploración y preprocesamiento del texto

### 2.1 Carga de datos y estadísticas básicas
""")

code(r"""
from datasets import load_dataset
import gensim.downloader as api

wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
news = load_dataset("fancyzhx/ag_news")
train_lines = wiki["train"]["text"]            # 1.8 M líneas (párrafos, encabezados y líneas vacías)
print({k: len(v) for k, v in wiki.items()}, "| AG News:", {k: len(v) for k, v in news.items()})
""")

md(r"""
### 2.2 Normalización y tokenizador propio

**Decisiones (y por qué):**

| Aspecto | Decisión | Justificación |
|---|---|---|
| Mayúsculas | todo a minúsculas | GloVe (`glove-wiki-gigaword-100`) es *uncased*; si no, `The` y `the` serían palabras distintas en nuestro vocabulario pero la misma en GloVe y la comparación no sería justa. |
| Marcadores de WikiText | `@-@`, `@,@`, `@.@` se **reunen** (`role @-@ playing` → `role-playing`, `1 @,@ 000` → `1,000`) | Son artefactos del dataset; GloVe sí tiene `role-playing` y `1,000` como tokens. |
| Puntuación | cada signo es un token independiente (`,` `.` `"` …) | GloVe también trata la puntuación como tokens; así las coberturas son comparables. |
| Números | **se conservan** tal cual (`1,000`, `3.5`, `2010`) | GloVe conserva los números; mapearlos a `<num>` rompería la correspondencia con su vocabulario. |
| Contracciones | estilo PTB como el de GloVe: `don't → do n't`, `it's → it 's` | Es la convención de Stanford (Wikipedia + Gigaword); WikiText ya trae `'s` separado. |
| Encabezados (` = Título = `) | se **descartan** del entrenamiento (se cuentan en las estadísticas) | No son oraciones; contaminarían las ventanas de contexto. |
| Tokens especiales | no se usan en SGNS (las palabras raras se **eliminan** como en word2vec). Para clasificación se usa `<unk>`/`<pad>` | Reemplazar raras por `<unk>` crearía un "supertoken" muy frecuente con contextos arbitrarios. |
| Artefactos de AG News | `\` → espacio, entidades HTML (`&lt;`) → carácter | Es ruido propio del dataset. |

El mismo `tokenize()` se aplica a WikiText, a AG News y (en la sección 6) a todo lo que entra al clasificador.
""")

code(r"""
# ---------------- Normalización + tokenizador propio ----------------
_WIKI_AT   = re.compile(r" @([-,.])@ ")                               # "role @-@ playing" -> "role-playing"
_NT        = re.compile(r"(\w)n't\b")                                  # don't -> do n't
_CLITIC    = re.compile(r"(\w)('s|'re|'ve|'ll|'d|'m)\b")               # it's  -> it 's
_TOKEN     = re.compile(r"n't|'(?:s|re|ve|ll|d|m)\b|\d+(?:[.,]\d+)*(?![^\W_])|[^\W_]+(?:-[^\W_]+)*|\S")

def normalize(text: str) -> str:
    text = html.unescape(text).replace("\\", " ").lower()
    text = _WIKI_AT.sub(r"\1", text)
    text = _NT.sub(r"\1 n't", text)
    text = _CLITIC.sub(r"\1 \2", text)
    return text

def tokenize(text: str) -> list:
    return _TOKEN.findall(normalize(text))

print(tokenize("The U.S. didn't spend $1,000 on role @-@ playing games; it's 3.5% of well-known 'Wall St.' data."))
""")

md(r"""
### 2.3 Comparación con NLTK (12 oraciones)
Se comparan los tokens (en minúsculas) de nuestro tokenizador contra `nltk.word_tokenize`. Se muestran solo los tokens que difieren.
""")

code(r"""
import nltk
from nltk.tokenize import word_tokenize
try: word_tokenize("test")
except LookupError: nltk.download("punkt_tab", quiet=True)

sents = [
    "I can't believe it's already 2026, isn't it?",
    "She said: \"Don't go!\" and left.",
    "The state-of-the-art model costs $1,250.50 per month.",
    "Dr. Smith works at the U.S. Department of Energy.",
    "They're well-known for their mother-in-law's cooking.",
    "We'll meet at 5:30 p.m. on Jan. 5th, okay?",
    "O'Brien's e-mail is john.doe@example.com.",
    "It's a 3-1 win; the score was 3.5 to 2.",
    "Wouldn't you've known? That's rock 'n' roll!",
    "COVID-19 cases rose 12% in New York (N.Y.) yesterday.",
    "The anti-virus wasn't updated, so e.g. spam got through.",
    "Prices rose from $1,000 to 1,500 euros -- a 50% jump.",
]
rows = []
for s in sents:
    mine, ref = tokenize(s), [t.lower() for t in word_tokenize(s)]
    only_m = [t for t in mine if t not in ref]; only_n = [t for t in ref if t not in mine]
    rows.append({"oración": s, "solo_mío": only_m, "solo_nltk": only_n, "igual": mine == ref})
cmp_df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 80, "display.width", 220)
display(cmp_df)
print(f"Coinciden exactamente: {cmp_df['igual'].sum()}/{len(cmp_df)}")
""")

md(r"""
**Resultado: coinciden exactamente en 3 de las 12 oraciones.** Coinciden en contracciones (`can't → ca n't`, `it's → it 's`, `isn't → is n't`), compuestos con guion (`state-of-the-art`, `mother-in-law`), y moneda/números (`$1,250.50`). Difieren en:
- **Abreviaturas con punto** (`Dr.`, `U.S.`, `p.m.`, `Jan.`, `N.Y.`, `e.g.`): NLTK las deja como un token (`dr.`, `u.s.`, `p.m.`); nuestro tokenizador las parte en letras y puntos (`u . s .`) porque una regla simple no distingue "punto de abreviatura" de "punto final". Es su principal debilidad: genera tokens ruidosos (`u`, `s`).
- **Comillas:** NLTK usa la convención Treebank (` `` ` y `''`); nosotros dejamos `"` como GloVe.
- **Números con separadores:** `5:30` y `3-1` son un token en NLTK; nosotros los partimos (`5 : 30`, `3 - 1`). Igual `--` (NLTK: un token; nosotros: dos `-`).
- **Apóstrofos internos:** `O'Brien` → NLTK un token; nosotros `o ' brien`. En `'n'`, NLTK da `'n`, nosotros `n`.
- **e-mails/URL:** NLTK separa solo parcialmente (`john.doe`, `example.com`); nosotros fragmentamos todo (`john doe example com`).

Para un corpus de ~100 M de tokens estas diferencias afectan a una fracción pequeña de los tokens; lo importante es que el **mismo** tokenizador se use en el corpus, en AG News y al consultar benchmarks.
""")

md(r"""
### 2.4 Tokenización de todo el corpus: tokens, líneas y artículos
Un solo pase sobre las 1.8 M líneas: se cuenta la frecuencia de cada palabra (corpus **completo**) y se guardan los IDs del **subconjunto** de entrenamiento (primeras ~30 M de tokens). Se cachea en `artifacts/`.
""")

code(r"""
HEAD_RE    = re.compile(r"^\s*=+ .* =+\s*$")          # encabezados " = X = ", " = = Y = = "
ARTICLE_RE = re.compile(r"^ = [^=].* = \n?$")          # solo nivel 1 => un artículo

stats_path, counts_path = ART / "corpus_stats.json", ART / "counts_full.pkl"
sub_ids_path, sub_len_path, vocab_path = ART / "sub_ids_dyn.npy", ART / "sub_line_len_dyn.npy", ART / "sub_vocab_dyn.json"

if not (stats_path.exists() and counts_path.exists() and sub_ids_path.exists()):
    t0 = time.time()
    counts = collections.Counter()
    dyn = {}                                  # palabra -> id dinámico (solo para el subconjunto)
    sub_ids, sub_len = [], []
    n_lines = n_empty = n_head = n_articles = n_tokens = 0
    in_subset = True
    for line in tqdm(train_lines, desc="tokenizando"):
        n_lines += 1
        if not line.strip(): n_empty += 1; continue
        if HEAD_RE.match(line):
            n_head += 1; n_articles += bool(ARTICLE_RE.match(line)); continue
        toks = tokenize(line)
        if not toks: continue
        counts.update(toks); n_tokens += len(toks)
        if in_subset:
            sub_ids.extend(dyn.setdefault(w, len(dyn)) for w in toks)
            sub_len.append(len(toks))
            if n_tokens >= SUBSET_TOKENS: in_subset = False
    stats = dict(n_lines=n_lines, n_empty=n_empty, n_headings=n_head, n_articles=n_articles,
                 n_text_lines=n_lines - n_empty - n_head, n_tokens=n_tokens, n_types=len(counts),
                 subset_tokens=len(sub_ids), subset_lines=len(sub_len), secs=time.time() - t0)
    np.save(sub_ids_path, np.asarray(sub_ids, dtype=np.int32)); np.save(sub_len_path, np.asarray(sub_len, dtype=np.int32))
    json.dump([w for w, _ in sorted(dyn.items(), key=lambda kv: kv[1])], open(vocab_path, "w", encoding="utf-8"))
    pickle.dump(dict(counts), open(counts_path, "wb")); json.dump(stats, open(stats_path, "w"), indent=1)

stats = json.load(open(stats_path)); counts_full = pickle.load(open(counts_path, "rb"))
print(json.dumps(stats, indent=1))
""")

md(r"""
### 2.5 Ley de Zipf
Frecuencia vs. rango en escala log-log, y cobertura acumulada de las palabras más frecuentes.
""")

code(r"""
freqs = np.sort(np.fromiter(counts_full.values(), dtype=np.int64))[::-1]
ranks = np.arange(1, len(freqs) + 1); total = freqs.sum()

# Ajuste de la pendiente en el tramo central (rangos 10 .. 10 000), donde Zipf suele valer
sl = slice(9, 10_000)
slope, intercept = np.polyfit(np.log(ranks[sl]), np.log(freqs[sl]), 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].loglog(ranks, freqs, lw=1.2, label="datos")
ax[0].loglog(ranks[sl], np.exp(intercept) * ranks[sl] ** slope, "r--", label=f"ajuste (pendiente={slope:.2f})")
ax[0].set(xlabel="rango", ylabel="frecuencia", title="Zipf: frecuencia vs rango (WikiText-103)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].semilogx(ranks, np.cumsum(freqs) / total * 100)
ax[1].set(xlabel="rango (top-k palabras)", ylabel="% de tokens cubiertos", title="Cobertura acumulada"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

cov = {k: freqs[:k].sum() / total * 100 for k in (10, 1_000, 30_000)}
print("Pendiente log-log (rangos 10-10 000): %.3f  (Zipf ideal ≈ -1)" % slope)
for k, v in cov.items(): print(f"  top {k:>6,} palabras cubren {v:5.2f}% de los tokens")
""")

md(r"""
**Lectura:** en el tramo central (rangos 10–10 000) la curva es casi una recta en log-log con pendiente **−1.07**, así que **la ley de Zipf se cumple aproximadamente**; se desvía en los extremos (la cola de palabras con frecuencia 1, 2, 3… forma "escalones"). El corpus tiene **97.5 M de tokens, 702 k tipos distintos, 29 444 artículos y 859 955 líneas de texto** (además de 636 k líneas vacías y 305 k encabezados descartados). Las **10 palabras más frecuentes cubren el 29.7 %** de los tokens, **las 1 000 más frecuentes el 68.5 %** y **las 30 000 más frecuentes el 95.4 %**: pocas palabras concentran casi todo el texto, lo que justifica el submuestreo de las frecuentes y el corte por `min_count`. (El total es menor que las ~103 M de tokens "oficiales" de WikiText-103 en parte porque aquí se unen los marcadores `@-@`/`@,@` y se descartan los encabezados.)
""")

md(r"""
### 2.6 Vocabulario con umbral de frecuencia mínima (`min_count`)
Tamaño de vocabulario y % de tokens desconocidos (tokens cuya palabra queda por debajo del umbral) sobre el corpus completo.
""")

code(r"""
rows = []
for mc in (1, 2, 3, 5, 10, 20, 50, 100):
    keep = freqs >= mc
    rows.append({"min_count": mc, "|V|": int(keep.sum()), "tokens cubiertos %": freqs[keep].sum() / total * 100,
                 "tokens desconocidos %": freqs[~keep].sum() / total * 100})
mc_df = pd.DataFrame(rows); display(mc_df.style.format({"tokens cubiertos %": "{:.3f}", "tokens desconocidos %": "{:.3f}", "|V|": "{:,}"}))
""")

md(r"""
**Lectura (corpus completo):** el vocabulario sin filtro es enorme (702 k tipos, probablemente inflado por números, compuestos con guion y palabras no latinas que el tokenizador conserva). Con `min_count=5` baja a **192 k** (−73 %) y solo el **0.81 %** de los tokens pasa a ser desconocido; con 50 queda en 51 k (2.9 % desconocidos) y con 100 en 33.8 k (4.2 %). Palabras con < 5 ocurrencias no tienen contextos suficientes para aprender un vector fiable, por eso se usa **`min_count = 5`**. Sobre el subconjunto de entrenamiento (primeros ~30 M de tokens) esto da |V| = **100 844** palabras y se descarta el 1.32 % de los tokens.

### 2.7 Vocabulario de entrenamiento, subconjunto y subsampling
Se construye el vocabulario **desde el subconjunto de ~30 M tokens** (`min_count=5`), ordenado por frecuencia (ID 0 = palabra más frecuente). Las palabras raras se eliminan del flujo, como hace word2vec.
""")

code(r"""
MIN_COUNT = 5
dyn_words = json.load(open(vocab_path, encoding="utf-8"))
dyn_ids   = np.load(sub_ids_path); dyn_len = np.load(sub_len_path)
dyn_counts = np.bincount(dyn_ids, minlength=len(dyn_words))

order = [i for i in np.argsort(-dyn_counts, kind="stable") if dyn_counts[i] >= MIN_COUNT]
words = [dyn_words[i] for i in order]                      # vocabulario ordenado por frecuencia
counts = dyn_counts[order].astype(np.int64)
remap = np.full(len(dyn_words), -1, dtype=np.int32); remap[order] = np.arange(len(order), dtype=np.int32)
w2i = {w: i for i, w in enumerate(words)}

new = remap[dyn_ids]; keep = new >= 0
line_of_tok = np.repeat(np.arange(len(dyn_len)), dyn_len)
train_ids = new[keep]                                       # flujo de IDs sin raras
line_len  = np.bincount(line_of_tok[keep], minlength=len(dyn_len)).astype(np.int64)
line_len  = line_len[line_len > 0]
V = len(words); N_TOK = len(train_ids)
print(f"Subconjunto: {N_TOK:,} tokens en {len(line_len):,} líneas | |V|={V:,} (min_count={MIN_COUNT}) | descartados por raros: {(~keep).mean()*100:.2f}%")
print("top-15:", words[:15])
""")

md(r"""
#### Submuestreo (subsampling) de palabras frecuentes (Mikolov et al.)
Cada ocurrencia de la palabra $w$ con frecuencia relativa $f(w)$ se **conserva** con probabilidad
$$P_{keep}(w)=\min\!\Big(1,\ \big(\sqrt{f(w)/t}+1\big)\,\frac{t}{f(w)}\Big)$$
con umbral $t$ (hiperparámetro). Las palabras muy frecuentes (*the, of, and, ,, .*) aportan poca información sobre su contexto y generan la mayoría de los pares de entrenamiento; descartarlas **acelera el entrenamiento** y deja más peso relativo a las palabras raras/informativas, lo que **mejora la calidad** de sus vectores.
""")

code(r"""
# (Los porcentajes de descarte son teóricos y empíricos; coinciden.)
def keep_prob(counts, t):
    f = counts / counts.sum()
    return np.minimum(1.0, (np.sqrt(f / t) + 1) * t / f)

rng = np.random.default_rng(SEED)
rows = []
for t in (1e-3, 1e-4, 1e-5):
    kp = keep_prob(counts, t)
    sampled = rng.random(N_TOK) < kp[train_ids]
    row = {"t": t, "tokens que sobreviven %": sampled.mean() * 100}
    for w in ("the", "of", "and"):
        row[f"'{w}' descartado %"] = (1 - kp[w2i[w]]) * 100          # probabilidad teórica
        m = train_ids == w2i[w]; row[f"'{w}' (empírico) %"] = (~sampled[m]).mean() * 100
    rows.append(row)
display(pd.DataFrame(rows).style.format({c: "{:.2f}" for c in pd.DataFrame(rows).columns if c != "t"}))
""")

md(r"""
**Resultado:** con $t=10^{-4}$ se descarta el **96.0 %** de las apariciones de *"the"*, el **93.7 %** de *"of"* y el **93.4 %** de *"and"* (con $t=10^{-3}$: 86.2 %, 77.7 %, 76.3 %; con $t=10^{-5}$: 98.8 %, 98.1 %, 98.0 %), y sobrevive el 54.6 % de los tokens totales. Los valores empíricos coinciden con la fórmula.

### 2.8 Pares (centro, contexto) del skip-gram
Con ventana $W$ (fija = máxima), cada posición genera hasta $2W$ pares; los pares **no cruzan líneas** (párrafos). Para una línea de longitud $L>W$ el total exacto es $2(LW-W(W+1)/2)$ y $L(L-1)$ si $L\le W$. Se reporta por epoch antes y después del submuestreo (con ventana dinámica de word2vec el promedio real es ≈ $(W+1)/2$ por lado).
""")

code(r"""
def n_pairs(lens, W):
    lens = lens.astype(np.int64)
    return int(np.where(lens > W, 2 * (lens * W - W * (W + 1) // 2), lens * (lens - 1)).sum())

def sub_line_lens(t, rng):
    kp = keep_prob(counts, t); m = rng.random(N_TOK) < kp[train_ids]
    line_idx = np.repeat(np.arange(len(line_len)), line_len)
    return np.bincount(line_idx[m], minlength=len(line_len)), m.sum()

rows = []
for W in (2, 5, 10):
    row = {"ventana W": W, "pares/epoch (sin subsampling)": n_pairs(line_len, W)}
    for t in (1e-3, 1e-4, 1e-5):
        sl_, kept = sub_line_lens(t, np.random.default_rng(SEED)); row[f"t={t:g}"] = n_pairs(sl_, W)
    rows.append(row)
pairs_df = pd.DataFrame(rows); display(pairs_df.style.format({c: "{:,}" for c in pairs_df.columns if c != "ventana W"}))
""")

md(r"""
**Resultado (ventana fija $W=5$, 29.6 M tokens):** el corpus produce **288.2 M pares por epoch sin submuestreo**; con submuestreo baja a **205.6 M** ($t=10^{-3}$), **153.9 M** ($t=10^{-4}$, −46.6 %) y **89.3 M** ($t=10^{-5}$, −69 %). Con la ventana dinámica de word2vec (radio aleatorio en $1..W$) el número efectivo de pares es ≈ 60 % del de la ventana fija. El submuestreo reduce el trabajo por epoch casi a la mitad sin perder las palabras informativas.
""")
