CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
### 4.4 Word2Vec de gensim (referencia) y GloVe preentrenado
**gensim:** `Word2Vec(sg=1, negative=K, sample=t, window=W, vector_size=d)` entrenado sobre **exactamente el mismo flujo de tokens** (mismas líneas, mismo subconjunto de ~29.6 M de tokens, vocabulario ya filtrado con `min_count=5`, por lo que el vocabulario resulta idéntico) y con los hiperparámetros de la **mejor configuración de nuestro SGNS**. Se evalúa al final de cada epoch con las mismas funciones. **GloVe:** `glove-wiki-gigaword-100` tal cual (400 000 palabras, 100 dim).
""")

code(r"""
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

words_arr = np.array(words, dtype=object)
class LineCorpus:
    # Iterable reiniciable: devuelve cada línea del subconjunto como lista de palabras (mismo flujo que usa el SGNS).
    def __init__(self, n_use, n_tok): self.n_use, self.n_tok = n_use, n_tok
    def __iter__(self):
        off = 0
        for L in line_len[:self.n_use]:
            yield words_arr[train_ids[off:off + L]].tolist(); off += int(L)

class EpochEval(CallbackAny2Vec):
    def __init__(self): self.hist, self.t0, self.prev_loss = [], None, 0.0
    def on_epoch_begin(self, model): self.t0 = time.time()
    def on_epoch_end(self, model):
        secs = time.time() - self.t0; cum = model.get_latest_training_loss(); loss_ep = cum - self.prev_loss; self.prev_loss = cum
        ws, E = model.wv.index_to_key, model.wv.vectors
        ana = analogy_eval(ws, E, methods=("add",)); sim = similarity_eval(ws, E, WS353)
        self.hist.append(dict(epoch=len(self.hist) + 1, loss_cum_epoch=loss_ep, ana_sem=ana["add"]["sem"], ana_syn=ana["add"]["syn"], ana_total=ana["add"]["total"],
                              ws353=sim["spearman"], neighbors=neighbors(ws, E), epoch_secs=secs))
        h = self.hist[-1]; print(f"[gensim] ep {h['epoch']} | ana tot {h['ana_total']*100:5.2f} (sem {h['ana_sem']*100:5.2f} syn {h['ana_syn']*100:5.2f}) | WS353 {h['ws353']:.3f} | {secs:.0f}s")

def train_gensim(name, d, W, K, t, epochs, workers=8, seed=SEED):
    path = ART / "runs" / f"{name}.json"
    if path.exists(): return json.load(open(path))
    cb = EpochEval(); t0 = time.time()
    m = Word2Vec(vector_size=d, window=W, negative=K, sample=t, min_count=MIN_COUNT, sg=1, hs=0, workers=workers, seed=seed, epochs=epochs, compute_loss=True)
    corpus = LineCorpus(len(line_len), N_TOK); m.build_vocab(corpus); assert len(m.wv) == V, (len(m.wv), V)
    m.train(corpus, total_examples=m.corpus_count, epochs=epochs, callbacks=[cb], compute_loss=True)
    total = time.time() - t0
    m.wv.save(str(ART / "runs" / f"{name}.kv"))
    out = dict(name=name, config=dict(d=d, W=W, K=K, t=t, epochs=epochs, workers=workers, tokens_corpus=N_TOK, V=len(m.wv)), history=cb.hist, total_secs=total,
               train_secs=sum(h["epoch_secs"] for h in cb.hist))
    json.dump(out, open(path, "w"), default=float); return out
""")

code(r"""
# Word2Vec de gensim con la configuración de la mejor corrida SGNS
g_res = train_gensim("gensim_best", d=best_cfg["d"], W=best_cfg["W"], K=best_cfg["K"], t=best_cfg["t"], epochs=best_cfg["epochs"])
gensim_kv = KeyedVectors.load(str(ART / "runs" / "gensim_best.kv"))
glove = api.load("glove-wiki-gigaword-100")                # GloVe preentrenado, sin entrenamiento adicional
print(f"gensim: {g_res['total_secs']/60:.1f} min | GloVe: {len(glove.key_to_index):,} palabras x {glove.vector_size}")
sgns_E = np.load(ART / "runs" / f"{best_name}.npy").astype(np.float32)
""")
