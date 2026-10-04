CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
## 4. Construcción y entrenamiento de los embeddings

### 4.1 Utilidades de evaluación (benchmarks de gensim)
- **Analogías** (`questions-words.txt`, 19 544 preguntas, 14 categorías: 5 semánticas + 9 sintácticas). *a es a b como c es a ?* se resuelve con
  **3CosAdd** $\arg\max_x \cos(x,b)-\cos(x,a)+\cos(x,c)$ y **3CosMul** $\arg\max_x \frac{\cos(x,b)\cos(x,c)}{\cos(x,a)+\varepsilon}$ (cosenos desplazados a $[0,1]$, $\varepsilon=10^{-3}$), excluyendo $a,b,c$ de los candidatos. Solo se evalúan las preguntas cuyas 4 palabras están en el espacio de búsqueda (se reporta la cobertura).
- **Similitud de palabras**: correlación de Spearman entre el coseno y la puntuación humana en WordSim-353 (y SimLex-999, solo al final).
- Todo se calcula en GPU sobre vectores normalizados.
""")

code(r"""
from gensim.test.utils import datapath
from scipy.stats import spearmanr

def load_pairs(fname):
    out = []
    for l in open(datapath(fname), encoding="utf-8"):
        if l.startswith("#") or not l.strip(): continue
        a, b, s = l.rstrip("\n").split("\t")[:3]; out.append((a.lower(), b.lower(), float(s)))
    return out

def load_analogies():
    cats, cur = {}, None
    for l in open(datapath("questions-words.txt"), encoding="utf-8"):
        if l.startswith(":"): cur = l[1:].strip(); cats[cur] = []
        else: cats[cur].append(tuple(l.lower().split()))
    return cats

WS353, SIMLEX = load_pairs("wordsim353.tsv"), load_pairs("simlex999.txt")
ANALOGIES = load_analogies(); CATS = list(ANALOGIES); SEM_CATS, SYN_CATS = CATS[:5], CATS[5:]
print(len(WS353), "pares WS353 |", len(SIMLEX), "pares SimLex |", sum(map(len, ANALOGIES.values())), "analogías en", len(CATS), "categorías")

CONTROL = ["king", "france", "computer", "good", "january", "run"]

@torch.no_grad()
def analogy_eval(words, E, restrict=30_000, allowed=None, methods=("add", "mul"), bs=256):
    # words: lista alineada con las filas de E. Espacio de búsqueda = primeras `restrict` filas, o la lista `allowed` (vocabulario compartido).
    if allowed is None: idx = np.arange(min(restrict, len(words)))
    else:
        pos = {w: i for i, w in enumerate(words)}; idx = np.array([pos[w] for w in allowed if w in pos])
    En = F.normalize(torch.as_tensor(E)[idx].float().to(DEVICE), dim=1)
    loc = {words[i]: j for j, i in enumerate(idx)}
    out = {m: {} for m in methods}; cover = {}
    for cat, qs in ANALOGIES.items():
        Q = [tuple(loc[x] for x in q) for q in qs if all(x in loc for x in q)]
        cover[cat] = (len(Q), len(qs))
        if not Q:
            for m in methods: out[m][cat] = float("nan")
            continue
        Q = torch.tensor(Q, device=DEVICE); hits = {m: 0 for m in methods}
        for s in range(0, len(Q), bs):
            q = Q[s:s + bs]; S = torch.einsum("nkd,vd->nkv", En[q[:, :3]], En)        # cos con a, b, c
            ca, cb, cc = S[:, 0], S[:, 1], S[:, 2]
            sc = {"add": cb - ca + cc, "mul": ((cb + 1) / 2) * ((cc + 1) / 2) / ((ca + 1) / 2 + 1e-3)}
            for m in methods:
                x = sc[m]; x.scatter_(1, q[:, :3], -1e9)                                # excluir a, b, c
                hits[m] += (x.argmax(1) == q[:, 3]).sum().item()
        for m in methods: out[m][cat] = hits[m] / len(Q)
    # sem / syn / total agrupando preguntas (como gensim)
    res = {}
    for m in methods:
        def pooled(cs):
            n = sum(cover[c][0] for c in cs); return sum(out[m][c] * cover[c][0] for c in cs if cover[c][0]) / max(n, 1)
        res[m] = dict(per_cat=out[m], sem=pooled(SEM_CATS), syn=pooled(SYN_CATS), total=pooled(CATS))
    res["coverage"] = dict(per_cat=cover, n_eval=sum(c[0] for c in cover.values()), n_total=sum(c[1] for c in cover.values()))
    return res

@torch.no_grad()
def similarity_eval(words, E, pairs):
    pos = {w: i for i, w in enumerate(words)}
    P = [(pos[a], pos[b], s) for a, b, s in pairs if a in pos and b in pos]
    En = F.normalize(torch.as_tensor(E).float().to(DEVICE), dim=1)
    ia, ib = torch.tensor([p[0] for p in P], device=DEVICE), torch.tensor([p[1] for p in P], device=DEVICE)
    cos = (En[ia] * En[ib]).sum(1).cpu().numpy()
    return dict(spearman=float(spearmanr(cos, [p[2] for p in P]).correlation), n_eval=len(P), n_total=len(pairs))

@torch.no_grad()
def neighbors(words, E, queries=CONTROL, k=5, restrict=None):
    pos = {w: i for i, w in enumerate(words)}
    En = F.normalize(torch.as_tensor(E).float().to(DEVICE), dim=1)
    En_s = En[:restrict] if restrict else En
    out = {}
    for q in queries:
        if q not in pos: out[q] = []; continue
        sims = En_s @ En[pos[q]]; top = sims.topk(k + 1).indices.tolist()
        out[q] = [(words[i], round(float(sims[i]), 3)) for i in top if i != pos[q]][:k]
    return out
""")

md(r"""
### 4.2 Skip-gram con negative sampling (SGNS) desde cero en PyTorch
**Modelo.** Dos tablas `nn.Embedding`: `inp` ($W$, vector de la palabra como **centro**) y `out` ($W'$, como **contexto**). Para un par (centro $w$, contexto $c$) y $K$ negativos $n_k\sim P_n(w)\propto f(w)^{3/4}$:
$$\mathcal L=-\log\sigma(u_c^\top v_w)-\sum_{k=1}^{K}\log\sigma(-u_{n_k}^\top v_w)$$

**Implementación (todo en GPU para que el entrenamiento sea rápido):**
1. En cada epoch se aplica el **submuestreo** (sección 2.7) sobre el flujo de IDs.
2. Se generan los pares con **ventana dinámica** (radio aleatorio $r\sim U\{1..W\}$ por posición, como word2vec) sin cruzar límites de línea.
3. Los pares se barajan por bloques y se procesan en mini-batches; los negativos salen de una tabla pre-calculada de la distribución unigram$^{3/4}$.
4. Gradientes **dispersos** (`sparse=True`) con `SparseAdam`; la tasa de aprendizaje decae linealmente hasta 5 % del valor inicial. Inicialización: `inp ~ U(-0.5/d, 0.5/d)`, `out = 0` (como word2vec).
""")

code(r"""
class SGNS(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.inp = nn.Embedding(V, d, sparse=True); self.out = nn.Embedding(V, d, sparse=True)
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d); nn.init.zeros_(self.out.weight)

    def forward(self, center, ctx, neg):                    # center,ctx: (B,)  neg: (B,K)
        v = self.inp(center)                                  # (B,d)
        pos = (v * self.out(ctx)).sum(-1)                     # (B,)
        ng = torch.bmm(self.out(neg), v.unsqueeze(-1)).squeeze(-1)   # (B,K)
        return -(F.logsigmoid(pos) + F.logsigmoid(-ng).sum(-1)).mean()

def keep_prob_t(cnt, t):
    f = cnt / cnt.sum(); f = f.clamp_min(1e-12)
    return torch.clamp((torch.sqrt(f / t) + 1) * t / f, max=1.0)

def pair_chunks(kept, klid, W, dynamic, chunk, gen):
    # Genera pares (centro, contexto) por bloques de `chunk` posiciones centrales, sin cruzar líneas.
    M = kept.numel(); nch = math.ceil(M / chunk)
    for ci in torch.randperm(nch, generator=gen, device="cpu").tolist():
        a, e = ci * chunk, min(M, (ci + 1) * chunk); lo, hi = max(0, a - W), min(M, e + W)
        seg, sl = kept[lo:hi], klid[lo:hi]; n = seg.numel(); cl, ch = a - lo, e - lo
        r = torch.randint(1, W + 1, (n,), device=seg.device, generator=None) if dynamic else torch.full((n,), W, device=seg.device)
        ar = torch.arange(n, device=seg.device); cs, os_ = [], []
        for d in range(1, W + 1):
            m = n - d
            if m <= 0: break
            same = sl[:m] == sl[d:]
            fwd = same & (r[:m] >= d) & (ar[:m] >= cl) & (ar[:m] < ch)               # centro i, contexto i+d
            bwd = same & (r[d:] >= d) & (ar[d:] >= cl) & (ar[d:] < ch)               # centro i+d, contexto i
            cs += [seg[:m][fwd], seg[d:][bwd]]; os_ += [seg[d:][fwd], seg[:m][bwd]]
        c, o = torch.cat(cs), torch.cat(os_); p = torch.randperm(c.numel(), device=c.device)
        yield c[p], o[p], ci, nch

def train_sgns(name, d=100, W=5, K=5, t=1e-4, epochs=5, frac=1.0, lr=0.01, B=16384, dynamic=True, seed=SEED, verbose=True):
    path = ART / "runs" / f"{name}.json"; path.parent.mkdir(exist_ok=True)
    if path.exists(): return json.load(open(path))
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    n_use = int(np.searchsorted(np.cumsum(line_len), frac * N_TOK)) + 1 if frac < 1 else len(line_len)   # cortar en frontera de línea
    n_tok = int(line_len[:n_use].sum())
    ids = torch.as_tensor(train_ids[:n_tok], dtype=torch.long, device=DEVICE)
    lid = torch.as_tensor(np.repeat(np.arange(n_use, dtype=np.int32), line_len[:n_use]), device=DEVICE)
    cnt = torch.bincount(ids, minlength=V).double()
    kp = keep_prob_t(cnt, t).float()
    TS = 1 << 24; pn = cnt ** 0.75; pn = pn / pn.sum()                                 # tabla de negativos ~ unigram^0.75
    table = torch.repeat_interleave(torch.arange(V, device=DEVICE), torch.round(pn * TS).long()).int(); TS = table.numel()

    model = SGNS(V, d).to(DEVICE); opt = torch.optim.SparseAdam(model.parameters(), lr=lr)
    torch.cuda.reset_peak_memory_stats(); hist, steps_log, step = [], [], 0; t_start = time.time()
    for ep in range(epochs):
        t0 = time.time(); mask = torch.rand(n_tok, device=DEVICE) < kp[ids]; kept, klid = ids[mask], lid[mask]
        loss_sum = torch.zeros((), device=DEVICE); n_batches = 0; n_pairs_ep = 0
        for k_chunk, (c, o, ci, nch) in enumerate(pair_chunks(kept, klid, W, dynamic, 2_000_000, gen)):
            n_pairs_ep += c.numel()
            # decaimiento lineal del lr (progreso global = epochs completos + fracción de bloques) hasta 5 % del inicial
            for g in opt.param_groups: g["lr"] = lr * max(0.05, 1 - (ep + k_chunk / nch) / epochs)
            for s in range(0, c.numel(), B):
                cb, ob = c[s:s + B], o[s:s + B]; neg = table[torch.randint(0, TS, (cb.numel(), K), device=DEVICE)].long()
                loss = model(cb, ob, neg); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                loss_sum += loss.detach(); n_batches += 1; step += 1
                if step % 200 == 0: steps_log.append((step, loss.item()))
        torch.cuda.synchronize(); ep_secs = time.time() - t0
        E = model.inp.weight.detach(); ana = analogy_eval(words, E, methods=("add",)); ws = similarity_eval(words, E, WS353)
        rec = dict(epoch=ep + 1, loss=(loss_sum / n_batches).item(), ana_sem=ana["add"]["sem"], ana_syn=ana["add"]["syn"], ana_total=ana["add"]["total"],
                   ws353=ws["spearman"], neighbors=neighbors(words, E), epoch_secs=ep_secs, pairs=n_pairs_ep, tokens_kept=int(kept.numel()))
        hist.append(rec)
        if verbose: print(f"[{name}] ep {ep+1}/{epochs} loss {rec['loss']:.4f} | ana tot {rec['ana_total']*100:5.2f} (sem {rec['ana_sem']*100:5.2f} syn {rec['ana_syn']*100:5.2f}) | WS353 {rec['ws353']:.3f} | {ep_secs:.0f}s | pares {n_pairs_ep/1e6:.0f}M")
    total_secs = time.time() - t_start; peak = torch.cuda.max_memory_allocated() / 2**20
    np.save(ART / "runs" / f"{name}.npy", model.inp.weight.detach().half().cpu().numpy())
    out = dict(name=name, config=dict(d=d, W=W, K=K, t=t, epochs=epochs, frac=frac, lr=lr, B=B, dynamic=dynamic, tokens_corpus=n_tok, V=V),
               history=hist, steps_log=steps_log, train_secs=sum(h["epoch_secs"] for h in hist), total_secs=total_secs, peak_gpu_mb=peak)
    json.dump(out, open(path, "w"), default=float); del model, opt, ids, lid; torch.cuda.empty_cache(); return out
""")
