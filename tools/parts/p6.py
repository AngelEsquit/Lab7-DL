CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
## 6. Pipeline end-to-end: clasificación de AG News
**texto → tokens → vectores → modelo → salida.** Se usa el **mismo `tokenize()` de la sección 2** (minúsculas, contracciones PTB, números conservados). Se separa el **10 % del train como validación (estratificado)**; el test oficial (7 600 noticias) solo se usa **una vez por modelo seleccionado**.

**Modelos:**
1. **Baseline:** TF-IDF (unigramas + bigramas) + regresión logística.
2. **Embeddings aleatorios:** `nn.EmbeddingBag(mean)` + MLP, tabla inicializada al azar y entrenada con la tarea (vocabulario = palabras con frecuencia ≥ 2 en el train; el resto → `<unk>`).
3. **SGNS propio / Word2Vec gensim / GloVe:** el mismo clasificador inicializado con cada tabla, en dos variantes: **(a) congelada** y **(b) con fine-tuning**. Los tokens fuera del vocabulario del embedding se **omiten** de la bolsa (se reporta su porcentaje).

Clasificador: `EmbeddingBag(mean, padding_idx=0)` → `Linear(100,256)` → ReLU → Dropout(0.3) → `Linear(256,4)`; Adam (lr 1e-3), batch 256, hasta 10 epochs (40 en las variantes **congeladas**, que solo entrenan el MLP y convergen más lento), se conserva el epoch con mejor F1-macro de validación.
""")

code(r"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

LABELS = news["train"].features["label"].names
tok_path = ART / "agnews_tokens.pkl"
if tok_path.exists(): ag_tr_tok, ag_te_tok = pickle.load(open(tok_path, "rb"))
else:
    ag_tr_tok = [tokenize(t) for t in tqdm(news["train"]["text"], desc="tokenizando AG train")]
    ag_te_tok = [tokenize(t) for t in tqdm(news["test"]["text"], desc="tokenizando AG test")]
    pickle.dump((ag_tr_tok, ag_te_tok), open(tok_path, "wb"))
y_all = np.array(news["train"]["label"]); y_te = np.array(news["test"]["label"])

idx_tr, idx_va = train_test_split(np.arange(len(y_all)), test_size=0.10, stratify=y_all, random_state=SEED)
tok_tr, tok_va = [ag_tr_tok[i] for i in idx_tr], [ag_tr_tok[i] for i in idx_va]; y_tr, y_va = y_all[idx_tr], y_all[idx_va]
lens = np.array([len(t) for t in ag_tr_tok])
print(f"train {len(tok_tr):,} | val {len(tok_va):,} | test {len(ag_te_tok):,} | clases {LABELS}")
print(f"longitud en tokens: media {lens.mean():.1f}, p95 {np.percentile(lens, 95):.0f}, máx {lens.max()}")
print("distribución val:", np.bincount(y_va) / len(y_va))
""")

code(r"""
# Tres conjuntos de embeddings (el mejor SGNS, gensim con su misma configuración y GloVe)
EMBS = {"SGNS": (list(words), sgns_E), "gensim": (list(gensim_kv.index_to_key), gensim_kv.vectors), "GloVe": (list(glove.index_to_key), glove.vectors)}

# Porcentaje de tokens de AG News fuera del vocabulario de cada embedding (train+test)
all_tok = [t for ts in (ag_tr_tok, ag_te_tok) for doc in ts for t in doc]
cnt_ag = collections.Counter(all_tok); n_ag = sum(cnt_ag.values())
oov_rows = []
for name, (ws, _) in EMBS.items():
    vs = set(ws); oov_tok = sum(c for w, c in cnt_ag.items() if w not in vs); oov_typ = sum(1 for w in cnt_ag if w not in vs)
    oov_rows.append({"embedding": name, "|V|": len(ws), "tokens AG fuera de vocab %": oov_tok / n_ag * 100, "tipos AG fuera de vocab %": oov_typ / len(cnt_ag) * 100})
oov_df = pd.DataFrame(oov_rows).set_index("embedding"); display(oov_df.style.format({"|V|": "{:,}", "tokens AG fuera de vocab %": "{:.2f}", "tipos AG fuera de vocab %": "{:.2f}"}))
print("Tokens totales AG News:", f"{n_ag:,}", "| tipos:", f"{len(cnt_ag):,}")
""")

code(r"""
MAXLEN = 128
def encode(tok_lists, w2id, unk_id=None):
    # IDs con fila 0 = padding y fila 1 = <unk>; los tokens fuera de vocab se omiten (o van a <unk> si unk_id).
    X = np.zeros((len(tok_lists), MAXLEN), dtype=np.int64)
    for i, toks in enumerate(tok_lists):
        ids = []
        for t in toks:
            j = w2id.get(t)
            if j is not None: ids.append(j)
            elif unk_id is not None: ids.append(unk_id)
        if not ids: ids = [1]
        ids = ids[:MAXLEN]; X[i, :len(ids)] = ids
    return torch.as_tensor(X, device=DEVICE)

def prf(y, p):
    P, R, F1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, p), precision=P, recall=R, f1=F1)

class BagMLP(nn.Module):
    def __init__(self, n_rows, dim, init=None, freeze=False, hidden=256, n_cls=4):
        super().__init__()
        if init is not None: self.emb = nn.EmbeddingBag.from_pretrained(torch.as_tensor(init, dtype=torch.float32), freeze=freeze, mode="mean", padding_idx=0)
        else: self.emb = nn.EmbeddingBag(n_rows, dim, mode="mean", padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, n_cls))
    def forward(self, x): return self.mlp(self.emb(x))

@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval(); out = [model(X[i:i + bs]) for i in range(0, len(X), bs)]; return torch.cat(out)

def fit_bag(Xtr, ytr, Xva, yva, n_rows, dim, init=None, freeze=False, epochs=10, lr=1e-3, bs=256, seed=SEED):
    torch.manual_seed(seed); model = BagMLP(n_rows, dim, init, freeze).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.Adam(params, lr=lr)
    ytr_t, yva_t = torch.as_tensor(ytr, device=DEVICE), torch.as_tensor(yva, device=DEVICE)
    hist, best, best_state = [], -1, None; t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr), device=DEVICE); tl = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; loss = F.cross_entropy(model(Xtr[b]), ytr_t[b]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item() * len(b)
        lv = predict(model, Xva); vloss = F.cross_entropy(lv, yva_t).item(); m = prf(yva, lv.argmax(1).cpu().numpy())
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(Xtr), val_loss=vloss, **{"val_" + k: v for k, v in m.items()}))
        if m["f1"] > best: best, best_epoch, best_state = m["f1"], ep + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    n_par = sum(p.numel() for p in model.parameters()); n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(model=model, hist=hist, best_epoch=best_epoch, val=hist[best_epoch - 1], secs=time.time() - t0, params=n_par, trainable=n_tr)
""")

code(r"""
def make_inputs(init, toks_tr, toks_va, toks_te):
    # Devuelve (Xtr, Xva, Xte, n_rows, init_matrix) para una inicialización dada.
    if init == "Aleatorio":
        c = collections.Counter(t for d in toks_tr for t in d); vocab = [w for w, n in c.items() if n >= 2]
        w2id = {w: i + 2 for i, w in enumerate(vocab)}; enc = lambda L: encode(L, w2id, unk_id=1)
        return enc(toks_tr), enc(toks_va), enc(toks_te), len(vocab) + 2, None
    ws, E = EMBS[init]; w2id = {w: i + 2 for i, w in enumerate(ws)}; enc = lambda L: encode(L, w2id)
    M = np.zeros((len(ws) + 2, E.shape[1]), dtype=np.float32); M[2:] = E
    return enc(toks_tr), enc(toks_va), enc(toks_te), M.shape[0], M

def run_variants(init, frac=1.0, seed=SEED, epochs=10):
    # Entrena las variantes de una inicialización y devuelve la mejor según validación (test se evalúa una sola vez).
    if frac < 1:
        sub, _ = train_test_split(np.arange(len(y_tr)), train_size=frac, stratify=y_tr, random_state=seed)
        ttr, ytr = [tok_tr[i] for i in sub], y_tr[sub]
    else: ttr, ytr = tok_tr, y_tr
    Xtr, Xva, Xte, n_rows, M = make_inputs(init, ttr, tok_va, ag_te_tok)
    dim = 100 if M is None else M.shape[1]
    variants = {"entrenado": dict()} if init == "Aleatorio" else {"congelado": dict(freeze=True), "fine-tuning": dict(freeze=False)}
    fits = {v: fit_bag(Xtr, ytr, Xva, y_va, n_rows, dim, init=M, epochs=epochs * (4 if kw.get('freeze') else 1), seed=seed, **kw) for v, kw in variants.items()}
    best_v = max(fits, key=lambda v: fits[v]["val"]["val_f1"]); lt = predict(fits[best_v]["model"], Xte); pred = lt.argmax(1).cpu().numpy()
    test = dict(**prf(y_te, pred), cm=confusion_matrix(y_te, pred), variant=best_v)
    summary = {v: {k: f[k] for k in ("hist", "best_epoch", "val", "secs", "params", "trainable")} for v, f in fits.items()}
    del fits; torch.cuda.empty_cache(); return dict(summary=summary, test=test)
""")

code(r"""
# ---- Baseline TF-IDF + regresión logística ----
def run_baseline(frac=1.0, seed=SEED):
    if frac < 1:
        sub, _ = train_test_split(np.arange(len(y_tr)), train_size=frac, stratify=y_tr, random_state=seed); ttr, ytr = [tok_tr[i] for i in sub], y_tr[sub]
    else: ttr, ytr = tok_tr, y_tr
    t0 = time.time(); vec = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(ttr); Xva, Xte = vec.transform(tok_va), vec.transform(ag_te_tok)
    best = None
    for C in (1, 10, 30):                                              # C se elige con validación
        clf = LogisticRegression(C=C, max_iter=300).fit(Xtr, ytr); m = prf(y_va, clf.predict(Xva))
        if best is None or m["f1"] > best[0]["f1"]: best = (m, clf, C)
    m, clf, C = best; pred = clf.predict(Xte)
    return dict(val=m, test=dict(**prf(y_te, pred), cm=confusion_matrix(y_te, pred), variant=f"C={C}"), secs=time.time() - t0, params=Xtr.shape[1] * 4 + 4, features=Xtr.shape[1])

cls_path = ART / "cls_results.pkl"
cls = pickle.load(open(cls_path, "rb")) if cls_path.exists() else {}
FRACS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
for frac in FRACS:
    if ("TF-IDF", frac) not in cls: cls[("TF-IDF", frac)] = run_baseline(frac)
    for init in ("Aleatorio", "SGNS", "gensim", "GloVe"):
        if (init, frac) not in cls:
            t0 = time.time(); cls[(init, frac)] = run_variants(init, frac); print(f"{init:10s} frac={frac:<5} -> {cls[(init, frac)]['test']['variant']:12s} | {time.time()-t0:.0f}s")
    pickle.dump(cls, open(cls_path, "wb"))
print("listo")
""")

md(r"""
### 6.1 Resultados en validación (100 % del train)
Accuracy, precisión, recall y F1 (macro) en validación, parámetros totales/entrenables y tiempo de entrenamiento de **todas** las variantes.
""")

code(r"""
rows = [{"modelo": "TF-IDF + LogReg", "variante": cls[("TF-IDF", 1.0)]["test"]["variant"], "acc": cls[("TF-IDF", 1.0)]["val"]["acc"], "precision": cls[("TF-IDF", 1.0)]["val"]["precision"],
         "recall": cls[("TF-IDF", 1.0)]["val"]["recall"], "F1": cls[("TF-IDF", 1.0)]["val"]["f1"], "params totales": cls[("TF-IDF", 1.0)]["params"], "entrenables": cls[("TF-IDF", 1.0)]["params"],
         "tiempo (s)": cls[("TF-IDF", 1.0)]["secs"], "mejor epoch": None}]
for init in ("Aleatorio", "SGNS", "gensim", "GloVe"):
    for v, s in cls[(init, 1.0)]["summary"].items():
        rows.append({"modelo": init, "variante": v, "acc": s["val"]["val_acc"], "precision": s["val"]["val_precision"], "recall": s["val"]["val_recall"], "F1": s["val"]["val_f1"],
                     "params totales": s["params"], "entrenables": s["trainable"], "tiempo (s)": s["secs"], "mejor epoch": s["best_epoch"]})
val_df = pd.DataFrame(rows); display(val_df.style.format({"acc": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "F1": "{:.4f}", "params totales": "{:,}", "entrenables": "{:,}", "tiempo (s)": "{:.0f}"}, na_rep="-"))

fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
for a, init in zip(ax, ("Aleatorio", "SGNS", "gensim", "GloVe")):
    for v, s in cls[(init, 1.0)]["summary"].items():
        e = [h["epoch"] for h in s["hist"]]; l, = a.plot(e, [h["train_loss"] for h in s["hist"]], label=f"{v} (train)")
        a.plot(e, [h["val_loss"] for h in s["hist"]], "--", color=l.get_color(), label=f"{v} (val)")
    a.set_title(init); a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0].set_ylabel("pérdida (cross-entropy)"); plt.tight_layout(); FIG.mkdir(exist_ok=True); plt.savefig(FIG / "cls_loss.png", dpi=150); plt.show()
""")

md(r"""
### 6.2 Evaluación final en test (una sola vez por modelo seleccionado) y SimLex-999
Para cada inicialización se toma la **mejor variante según F1 de validación** y se evalúa una única vez sobre el test oficial. También se reporta la correlación de Spearman en **SimLex-999** (reservado para esta etapa) de los tres conjuntos de embeddings.
""")

code(r"""
rows = [{"modelo": "TF-IDF + LogReg", "variante": cls[("TF-IDF", 1.0)]["test"]["variant"], **{k: cls[("TF-IDF", 1.0)]["test"][k] for k in ("acc", "precision", "recall", "f1")}}]
for init in ("Aleatorio", "SGNS", "gensim", "GloVe"):
    t = cls[(init, 1.0)]["test"]; rows.append({"modelo": init, "variante": t["variant"], **{k: t[k] for k in ("acc", "precision", "recall", "f1")}})
test_df = pd.DataFrame(rows); display(test_df.style.format({k: "{:.4f}" for k in ("acc", "precision", "recall", "f1")}))

fig, ax = plt.subplots(1, 5, figsize=(21, 4))
names = ["TF-IDF", "Aleatorio", "SGNS", "gensim", "GloVe"]
for a, n in zip(ax, names):
    cm = cls[(n, 1.0)]["test"]["cm"]; a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > .5 else "black", fontsize=8)
    a.set_xticks(range(4)); a.set_xticklabels(LABELS, rotation=45); a.set_yticks(range(4)); a.set_yticklabels(LABELS)
    a.set_title(f"{n} ({cls[(n, 1.0)]['test']['variant']})  F1={cls[(n, 1.0)]['test']['f1']:.3f}", fontsize=9); a.set_xlabel("predicho"); a.set_ylabel("real")
plt.tight_layout(); FIG.mkdir(exist_ok=True); plt.savefig(FIG / "cls_confusion.png", dpi=150); plt.show()

simlex_rows = [{"embedding": n, "Spearman SimLex-999": (r := similarity_eval(ws, E, SIMLEX))["spearman"], "pares evaluados": f"{r['n_eval']}/{r['n_total']}"} for n, (ws, E) in EMBS.items()]
simlex_df = pd.DataFrame(simlex_rows).set_index("embedding"); display(simlex_df.style.format({"Spearman SimLex-999": "{:.3f}"}))
""")

md(r"""
### 6.3 F1 de test vs. fracción de datos de entrenamiento de AG News
Misma metodología para fracciones del train (1 %, 5 %, 10 %, 25 %, 50 %, 100 %; la validación completa se usa para elegir la variante y el test se evalúa una vez por punto).
""")

code(r"""
frac_rows = [{"fracción": f, "modelo": n, "variante": cls[(n, f)]["test"]["variant"], "F1 test": cls[(n, f)]["test"]["f1"], "acc test": cls[(n, f)]["test"]["acc"]} for f in FRACS for n in ["TF-IDF", "Aleatorio", "SGNS", "gensim", "GloVe"]]
frac_df = pd.DataFrame(frac_rows); display(frac_df.pivot(index="fracción", columns="modelo", values="F1 test").style.format("{:.4f}"))
plt.figure(figsize=(7, 4.5))
for n, mk in zip(["TF-IDF", "Aleatorio", "SGNS", "gensim", "GloVe"], "sov^D"):
    d = frac_df[frac_df["modelo"] == n]; plt.plot(d["fracción"] * 100, d["F1 test"], marker=mk, label=n)
plt.xscale("log"); plt.xlabel("% del train de AG News"); plt.ylabel("F1 macro (test)"); plt.title("F1 de test vs. fracción de datos"); plt.grid(alpha=.3, which="both"); plt.legend(); plt.tight_layout(); FIG.mkdir(exist_ok=True); plt.savefig(FIG / "f1_vs_fraccion.png", dpi=150); plt.show()
""")
