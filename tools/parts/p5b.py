CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

code(r"""
EMB = {"SGNS": Emb("SGNS", words, sgns_E), "gensim": Emb("gensim", gensim_kv.index_to_key, gensim_kv.vectors), "GloVe": Emb("GloVe", glove.index_to_key, glove.vectors)}

# 1) Verificación: nuestra analogy() vs most_similar() de gensim (200 preguntas aleatorias de questions-words, vocabulario completo)
rows = []
for n, e in EMB.items():
    t1, t5, md_ = check_against_gensim(e); rows.append({"modelo": n, "top-1 coincide %": t1 * 100, "top-5 (conjunto) coincide %": t5 * 100, "máx |Δ similitud|": md_})
display(pd.DataFrame(rows).set_index("modelo").style.format({"top-1 coincide %": "{:.1f}", "top-5 (conjunto) coincide %": "{:.1f}", "máx |Δ similitud|": "{:.1e}"}))
""")

md(r"""
**Verificación:** nuestra función `analogy` reproduce a `most_similar` de gensim (misma respuesta y similitudes idénticas salvo error de punto flotante), así que la implementación de 3CosAdd es correcta.
""")

code(r"""
# 2) Tablas de las 30 analogías propias, con exclusión de a, b, c (top-5 con similitud, rango y coseno de la respuesta correcta)
ind = {n: individual_table(e, exclude=True) for n, e in EMB.items()}
summ = []
for n, df in ind.items():
    d = df.dropna(subset=["rango"])
    for typ, g in d.groupby("tipo", sort=False): summ.append({"modelo": n, "tipo": typ, "top-1 %": (g["rango"] == 1).mean() * 100, "top-5 %": (g["rango"] <= 5).mean() * 100, "rango mediano": g["rango"].median(), "cos(correcta)": g["cos_correcta"].mean()})
    summ.append({"modelo": n, "tipo": "TOTAL (%d)" % len(d), "top-1 %": (d["rango"] == 1).mean() * 100, "top-5 %": (d["rango"] <= 5).mean() * 100, "rango mediano": d["rango"].median(), "cos(correcta)": d["cos_correcta"].mean()})
summ = pd.DataFrame(summ)
display(summ.style.format({"top-1 %": "{:.0f}", "top-5 %": "{:.0f}", "rango mediano": "{:.0f}", "cos(correcta)": "{:.3f}"}).hide(axis="index"))
""")

code(r"""
pd.set_option("display.max_rows", 200, "display.width", 250, "display.max_colwidth", 95)
for n in EMB:
    print(f"\n===== {n} =====")
    display(ind[n][["tipo", "analogía", "esperada", "top5", "rango", "cos_correcta"]].style.format({"cos_correcta": "{:.3f}", "rango": "{:.0f}"}, na_rep="-").hide(axis="index"))
""")

md(r"""
#### Repetición **sin excluir** las palabras de la consulta
Sin la exclusión, el vecino más cercano del vector $\hat b-\hat a+\hat c$ suele ser una de las propias palabras de la consulta. Se cuenta cuántas veces el primer lugar es $b$, $c$, $a$ o la respuesta correcta $d$.
""")

code(r"""
rows, examples = [], {}
for n, e in EMB.items():
    cnt = collections.Counter(); tot = 0
    for typ, qs in MY_ANALOGIES.items():
        for a, b, c, d in qs:
            if not all(x in e.w2i for x in (a, b, c, d)): continue
            top, _ = e.analogy(a, b, c, 5, exclude=False); w = top[0][0]; tot += 1
            cnt["b (2.º argumento)"] += w == b; cnt["c (3.er argumento)"] += w == c; cnt["a"] += w == a; cnt["d (correcta)"] += w == d
            cnt["otra palabra"] += w not in (a, b, c, d)
            if (a, b, c) == ("man", "woman", "king"): examples[n] = top
    rows.append({"modelo": n, "n": tot, **{k: v / tot * 100 for k, v in cnt.items()}})
display(pd.DataFrame(rows).set_index("modelo").style.format("{:.0f}", subset=pd.IndexSlice[:, ["b (2.º argumento)", "c (3.er argumento)", "a", "d (correcta)", "otra palabra"]]))
for n, top in examples.items(): print(f"{n:7s} man:woman::king:?  sin exclusión ->", ", ".join(f"{w} ({s:.2f})" for w, s in top))
""")

code(r"""
# 5.2 Evaluación sistemática: vocabulario compartido (30 000 más frecuentes en los tres modelos; orden = frecuencia en nuestro corpus)
shared = [w for w in words if w in glove.key_to_index][:30_000]; shared_set = set(shared)
print("tamaño del vocabulario compartido:", len(shared), "| ejemplo:", shared[:8])
sysres = {n: analogy_eval(e.words, e.En, allowed=shared, methods=("add", "mul")) for n, e in EMB.items()}
cov = sysres["SGNS"]["coverage"]; assert all(sysres[n]["coverage"]["per_cat"] == cov["per_cat"] for n in EMB)   # mismas preguntas en los tres
print(f"cobertura: {cov['n_eval']:,} / {cov['n_total']:,} preguntas evaluadas ({cov['n_eval'] / cov['n_total'] * 100:.1f} %) — idénticas en los tres modelos")

rows = []
for cat in CATS + ["semántica", "sintáctica", "TOTAL"]:
    r = {"categoría": cat}
    if cat in CATS: r["cobertura"] = f"{cov['per_cat'][cat][0]}/{cov['per_cat'][cat][1]}"
    else: r["cobertura"] = ""
    for n in EMB:
        for m, lab in (("add", "3CosAdd"), ("mul", "3CosMul")):
            r[f"{n} {lab}"] = sysres[n][m]["per_cat"][cat] * 100 if cat in CATS else sysres[n][m][{"semántica": "sem", "sintáctica": "syn", "TOTAL": "total"}[cat]] * 100
    rows.append(r)
sys_df = pd.DataFrame(rows).set_index("categoría"); display(sys_df.style.format({c: "{:.1f}" for c in sys_df.columns if c != "cobertura"}, na_rep="-"))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(17, 4.5), sharey=True)
for a, (m, lab) in zip(ax, (("add", "3CosAdd"), ("mul", "3CosMul"))):
    x = np.arange(len(CATS)); w = 0.27
    for k, n in enumerate(EMB): a.bar(x + (k - 1) * w, [sysres[n][m]["per_cat"][c] * 100 for c in CATS], w, label=n)
    a.set_xticks(x); a.set_xticklabels([c.replace("gram", "g") for c in CATS], rotation=60, ha="right", fontsize=8); a.set_title(f"Accuracy por categoría — {lab} (vocab. compartido 30k)"); a.grid(alpha=.3, axis="y"); a.legend()
ax[0].set_ylabel("accuracy (%)"); plt.tight_layout(); FIG.mkdir(exist_ok=True); plt.savefig(FIG / "analogias_categoria.png", dpi=150); plt.show()
""")

code(r"""
# Paralelismo de los vectores diferencia (b - a) dentro de cada categoría, vocabulario compartido
par = {n: parallelism(e, restrict_words=shared_set) for n, e in EMB.items()}
par_df = pd.DataFrame(par); par_df.index = [i if not i.startswith("_") else {"_aleatorio": "(pares aleatorios)", "_promedio": "PROMEDIO categorías"}[i] for i in par_df.index]
display(par_df.style.format("{:.3f}"))
""")

md(r"""
### t-SNE alrededor de ~500 palabras de grupos temáticos
Se proyectan a 2D (t-SNE sobre vectores normalizados, perplejidad 30) palabras de 10 grupos temáticos para **nuestro mejor SGNS** y, como referencia, para GloVe. Además se mide cuantitativamente la separación de grupos con el **coeficiente de silueta (coseno)** en el espacio original de 100/300 dimensiones.
""")

code(r"""
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

GROUPS = {
 "países": "france germany italy spain japan china russia india brazil canada australia mexico egypt greece turkey sweden norway poland ireland portugal argentina chile peru iran iraq korea vietnam thailand kenya nigeria ukraine austria belgium denmark finland switzerland netherlands cuba israel pakistan indonesia colombia venezuela scotland wales england hungary romania bulgaria serbia croatia syria jordan morocco".split(),
 "números": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth half double triple".split(),
 "animales": "dog cat horse cow pig sheep bird fish lion tiger bear wolf fox rabbit mouse elephant monkey snake eagle whale shark deer duck goat chicken frog turtle spider bee butterfly ant cattle donkey camel zebra giraffe leopard cheetah owl hawk crow dolphin seal squirrel bat rat".split(),
 "verbos": "run walk talk eat drink sleep write read play work build break open close buy sell speak sing dance swim jump climb drive fly cook wash clean carry throw catch push pull cut drop kill help teach learn study think believe remember forget".split(),
 "colores": "red blue green yellow black white orange purple pink brown grey gray gold silver golden crimson violet scarlet".split(),
 "profesiones": "doctor teacher lawyer engineer nurse farmer soldier pilot actor singer writer painter judge priest minister king queen president mayor chef scientist professor author journalist businessman banker police sailor driver artist musician".split(),
 "ciudades": "paris london berlin rome madrid tokyo moscow beijing cairo vienna athens lisbon dublin oslo stockholm warsaw prague budapest istanbul delhi bombay sydney toronto chicago boston seattle houston denver detroit atlanta dallas miami philadelphia amsterdam brussels geneva zurich munich barcelona milan venice florence naples".split(),
 "tiempo": "january february march april may june july august september october november december monday tuesday wednesday thursday friday saturday sunday spring summer autumn winter morning evening night".split(),
 "familia": "father mother son daughter brother sister uncle aunt nephew niece husband wife grandfather grandmother parents children cousin grandson granddaughter boy girl man woman baby".split(),
 "cuerpo": "head hand arm leg foot eye ear nose mouth heart brain finger hair skin face neck shoulder knee back blood bone tooth tongue lip".split(),
 "comida": "bread cheese milk meat rice sugar salt butter apple orange banana potato tomato beef pork chicken egg wine beer coffee tea soup cake chocolate fruit vegetable corn wheat honey".split(),
 "deportes": "football baseball basketball soccer tennis golf hockey rugby cricket boxing swimming cycling skiing wrestling volleyball athletics racing sailing rowing".split(),
 "transporte": "car bus train plane ship boat truck bicycle motorcycle taxi airplane helicopter subway tram ferry rocket yacht carriage wagon railway".split(),
 "naturaleza": "river mountain lake ocean sea forest desert island tree flower grass rain snow wind sun moon star cloud fire earth valley hill beach".split(),
 "ropa": "shirt dress hat shoes coat jacket skirt trousers glove scarf boots sock suit uniform".split(),
}
GROUPS["países"] += "luxembourg slovakia slovenia bolivia ecuador uruguay paraguay afghanistan nepal bangladesh malaysia philippines singapore ethiopia ghana tanzania uganda algeria libya tunisia".split()
common = lambda w: w in EMB["SGNS"].w2i and w in EMB["GloVe"].w2i
sel, lab = [], []
for g, ws_ in GROUPS.items():
    ws_ = [w for w in dict.fromkeys(ws_) if common(w)]; sel += ws_; lab += [g] * len(ws_)
print(f"{len(sel)} palabras en {len(GROUPS)} grupos:", {g: lab.count(g) for g in GROUPS})

def tsne_plot(ax, emb, title):
    X = emb.En[[emb.w2i[w] for w in sel]].cpu().numpy()
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED, metric="cosine").fit_transform(X)
    sil = silhouette_score(X, lab, metric="cosine")
    cmap = plt.get_cmap("tab20")
    for k, g in enumerate(GROUPS):
        m = np.array(lab) == g; ax.scatter(Z[m, 0], Z[m, 1], s=14, color=cmap(k), label=g)
        for z, w in zip(Z[m], np.array(sel)[m]): ax.annotate(w, z, fontsize=5.5, alpha=.8)
    ax.set_title(f"{title} (silueta coseno = {sil:.3f})"); ax.set_xticks([]); ax.set_yticks([]); return sil
fig, ax = plt.subplots(1, 2, figsize=(20, 9)); sils = {"SGNS": tsne_plot(ax[0], EMB["SGNS"], f"SGNS propio ({best_name})"), "GloVe": tsne_plot(ax[1], EMB["GloVe"], "GloVe-100")}
ax[0].legend(loc="best", fontsize=7, markerscale=1.5, ncol=2); plt.tight_layout(); FIG.mkdir(exist_ok=True); plt.savefig(FIG / "tsne.png", dpi=150); plt.show(); print("silueta:", sils)
""")
