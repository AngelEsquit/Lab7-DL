CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
## 5. Verificación de la aritmética vectorial
Se comparan **tres conjuntos de embeddings**: nuestro SGNS (mejor configuración), el Word2Vec de gensim (misma configuración y mismo corpus) y GloVe-100 preentrenado.

### 5.1 Analogías individuales
`analogy(a, b, c, k)` resuelve «a es a b como c es a ?» con **3CosAdd** sobre vectores normalizados: $\arg\max_x \cos(x,\ \hat b-\hat a+\hat c)$, excluyendo $a$, $b$ y $c$. Primero se verifica que coincide con `most_similar(positive=[b,c], negative=[a])` de gensim.
""")

code(r"""
from gensim.models import KeyedVectors

class Emb:
    # Envuelve (palabras, matriz) con vectores normalizados en GPU para consultas rápidas.
    def __init__(self, name, words, E):
        self.name, self.words = name, list(words); self.w2i = {w: i for i, w in enumerate(self.words)}
        self.En = F.normalize(torch.as_tensor(np.asarray(E, dtype=np.float32)).to(DEVICE), dim=1)
    def vec(self, w): return self.En[self.w2i[w]]
    @torch.no_grad()
    def analogy(self, a, b, c, k=5, exclude=True):
        q = self.vec(b) - self.vec(a) + self.vec(c)                      # a:b :: c:?   ->  b - a + c
        sims = self.En @ F.normalize(q, dim=0)                             # coseno con el vector resultante
        if exclude: sims[[self.w2i[a], self.w2i[b], self.w2i[c]]] = -2
        top = sims.topk(k); return [(self.words[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())], sims
    def rank_of(self, sims, w): return int((sims > sims[self.w2i[w]]).sum().item()) + 1
    def to_kv(self):
        kv = KeyedVectors(self.En.shape[1]); kv.add_vectors(self.words, self.En.cpu().numpy()); return kv

# Verificación contra gensim: mismas 200 analogías aleatorias, vocabulario completo
def check_against_gensim(emb, n=200, seed=0):
    kv = emb.to_kv(); rng = random.Random(seed)
    qs = [q for qs_ in ANALOGIES.values() for q in qs_ if all(x in emb.w2i for x in q)]
    top1 = top5set = 0; sample = rng.sample(qs, n); maxdiff = 0.0
    for a, b, c, d in sample:
        mine, _ = emb.analogy(a, b, c, k=5); ref = kv.most_similar(positive=[b, c], negative=[a], topn=5)
        top1 += mine[0][0] == ref[0][0]; top5set += {w for w, _ in mine} == {w for w, _ in ref}
        maxdiff = max(maxdiff, max(abs(m[1] - r[1]) for m, r in zip(mine, ref)))
    return top1 / n, top5set / n, maxdiff
""")

md(r"""
### Analogías propias (27, 7 tipos)
Para cada modelo se reporta el **top-5 con similitudes**, el **rango** de la respuesta correcta (1 = acierta) y su **similitud coseno con el vector resultante**.
""")

code(r"""
MY_ANALOGIES = {
  "género":           [("man","woman","king","queen"), ("boy","girl","brother","sister"), ("he","she","his","her"), ("father","mother","son","daughter"), ("uncle","aunt","nephew","niece")],
  "país-capital":     [("france","paris","germany","berlin"), ("japan","tokyo","italy","rome"), ("spain","madrid","egypt","cairo"), ("china","beijing","russia","moscow")],
  "país-gentilicio":  [("france","french","germany","german"), ("spain","spanish","japan","japanese"), ("italy","italian","china","chinese")],
  "comparativos":     [("good","better","bad","worse"), ("big","bigger","small","smaller"), ("fast","faster","slow","slower"), ("large","larger","tall","taller")],
  "superlativos":     [("good","best","bad","worst"), ("big","biggest","small","smallest"), ("fast","fastest","slow","slowest")],
  "tiempos verbales": [("walk","walked","run","ran"), ("go","went","see","saw"), ("write","wrote","take","took"), ("play","played","work","worked")],
  "plurales":         [("cat","cats","dog","dogs"), ("child","children","man","men"), ("city","cities","country","countries"), ("house","houses","book","books")],
}

def individual_table(emb, exclude=True):
    rows = []
    for typ, qs in MY_ANALOGIES.items():
        for a, b, c, d in qs:
            if not all(x in emb.w2i for x in (a, b, c, d)): rows.append(dict(tipo=typ, analogía=f"{a}:{b}::{c}:?", esperada=d, top5="(fuera de vocab)", rango=None, cos_correcta=None)); continue
            top, sims = emb.analogy(a, b, c, 5, exclude)
            rows.append(dict(tipo=typ, analogía=f"{a}:{b}::{c}:?", esperada=d, top5=", ".join(f"{w} ({s:.2f})" for w, s in top),
                             rango=emb.rank_of(sims, d), cos_correcta=float(sims[emb.w2i[d]])))
    return pd.DataFrame(rows)
""")

md(r"""
### 5.2 Evaluación sistemática sobre `questions-words.txt`
Vocabulario compartido: las **30 000 palabras más frecuentes presentes en los tres modelos**. Como cada modelo tiene su propio orden de frecuencia, se toma la **intersección** de los tres vocabularios y se ordena por la frecuencia en **nuestro corpus** (referencia común de los dos modelos entrenados), quedándose con las 30 000 primeras. Así las preguntas evaluadas son exactamente las mismas para todos.
""")

md(r"""
### Análisis de paralelismo (vectores diferencia)
Si la aritmética funcionara de forma exacta, todos los pares de una misma relación (p. ej. `paris−france`, `berlin−germany`, …) tendrían el **mismo vector diferencia**. Se mide la **similitud coseno promedio entre los vectores diferencia** $(\hat b-\hat a)$ de pares distintos de una misma categoría de `questions-words.txt` (y, como línea base, entre pares de palabras aleatorias).
""")

code(r"""
@torch.no_grad()
def parallelism(emb, restrict_words=None, max_pairs=300, seed=0):
    rng = random.Random(seed); res = {}; ok = lambda w: w in emb.w2i and (restrict_words is None or w in restrict_words)
    for cat, qs in ANALOGIES.items():
        pairs = list({(q[0], q[1]) for q in qs if ok(q[0]) and ok(q[1])} | {(q[2], q[3]) for q in qs if ok(q[2]) and ok(q[3])})
        if len(pairs) < 3: res[cat] = float("nan"); continue
        pairs = rng.sample(pairs, min(max_pairs, len(pairs)))
        D = F.normalize(torch.stack([emb.vec(b) - emb.vec(a) for a, b in pairs]), dim=1); S = D @ D.T
        n = len(pairs); res[cat] = float((S.sum() - n) / (n * (n - 1)))
    pool = [w for w in emb.words[:30000]]; r = torch.stack([emb.vec(rng.choice(pool)) - emb.vec(rng.choice(pool)) for _ in range(500)])
    D = F.normalize(r, dim=1); S = D @ D.T; res["_aleatorio"] = float((S.sum() - 500) / (500 * 499))
    res["_promedio"] = float(np.nanmean([v for k, v in res.items() if not k.startswith("_")]))
    return res
""")
