CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
## 7. Comparación de resultados
Tabla comparativa de los tres conjuntos de embeddings y gráficas. Las tablas también se exportan a `informe/*.tex` para incluirlas directamente en el informe de LaTeX.
""")

code(r"""
INF = Path("informe"); INF.mkdir(exist_ok=True)

def tex_table(header, rows, colfmt, **_):
    # Genera solo el entorno tabular de LaTeX (booktabs); el informe lo envuelve en su propio float / resizebox.
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("≈", r"$\approx$")
    out = [r"\begin{tabular}{" + colfmt + "}", r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    return chr(10).join(out)

# ---- Todas las corridas SGNS evaluadas sobre el MISMO vocabulario compartido (comparables con GloVe y gensim) ----
run_shared = {}
for name, r in results.items():
    E = np.load(ART / "runs" / f"{name}.npy").astype(np.float32); run_shared[name] = analogy_eval(words, E, allowed=shared, methods=("add", "mul"))
run_shared["gensim_best"] = sysres["gensim"]; glove_ref = sysres["GloVe"]["add"]["total"] * 100
tok_df = pd.DataFrame([{"corrida": n, "tokens": results[n]["config"]["tokens_corpus"], "d": results[n]["config"]["d"], "ana total % (vocab. compartido, 3CosAdd)": run_shared[n]["add"]["total"] * 100,
                        "3CosMul": run_shared[n]["mul"]["total"] * 100} for n in results]).set_index("corrida")
display(tok_df.style.format({"tokens": "{:,}", "ana total % (vocab. compartido, 3CosAdd)": "{:.1f}", "3CosMul": "{:.1f}"}))

fig, ax = plt.subplots(figsize=(6.2, 4.4))
fr = ["frac25", "frac50", "d100_base"]
ax.plot([results[n]["config"]["tokens_corpus"] for n in fr], [run_shared[n]["add"]["total"] * 100 for n in fr], "o-", lw=2, label="SGNS d=100 (25 %, 50 %, 100 % del corpus)")
for n, mk in (("d50", "s"), ("d300", "^"), ("W2", "v"), ("W10", "<"), ("K15", ">"), ("t1e-5", "D")):
    ax.scatter(results[n]["config"]["tokens_corpus"], run_shared[n]["add"]["total"] * 100, marker=mk, s=45, zorder=3, label=n)
ax.scatter(N_TOK, run_shared["gensim_best"]["add"]["total"] * 100, marker="*", s=130, c="k", zorder=4, label="Word2Vec gensim")
ax.scatter(6e9, glove_ref, marker="P", s=110, c="crimson", zorder=4, label=f"GloVe-100 (6 000 M tokens): {glove_ref:.1f} %")
ax.axhline(glove_ref, color="crimson", ls="--", lw=1)
ax.set_xscale("log"); ax.set_xlabel("tokens del corpus de entrenamiento (escala log)"); ax.set_ylabel("analogías, total (%)")
ax.set_title("Analogías (3CosAdd, vocab. compartido) vs. tamaño del corpus"); ax.grid(alpha=.3, which="both")
ax.legend(fontsize=6.5, loc="lower right", ncol=2); plt.tight_layout(); plt.savefig(FIG / "analogias_vs_tokens.png", dpi=150, bbox_inches="tight"); plt.show()
""")

code(r"""
# ---- Tabla comparativa SGNS vs gensim vs GloVe ----
ws_sims = {n: similarity_eval(e.words, e.En, WS353) for n, e in EMB.items()}
sl_sims = {n: similarity_eval(e.words, e.En, SIMLEX) for n, e in EMB.items()}
sgns_train = sum(h["epoch_secs"] for h in results[best_name]["history"]); sgns_ep = sgns_train / len(results[best_name]["history"])
cmp_rows = {
 "Tokens del corpus de entrenamiento": {"SGNS": f"{N_TOK:,}", "gensim": f"{N_TOK:,}", "GloVe": "6 000 000 000"},
 "Tamaño del vocabulario": {n: f"{len(e.words):,}" for n, e in EMB.items()},
 "Dimensión": {n: str(e.En.shape[1]) for n, e in EMB.items()},
 "Tiempo de entrenamiento": {"SGNS": f"{sgns_train/60:.1f} min ({sgns_ep:.0f} s/epoch x {len(results[best_name]['history'])})", "gensim": f"{g_res['train_secs']/60:.1f} min ({g_res['train_secs']/len(g_res['history']):.0f} s/epoch x {len(g_res['history'])})",
                             "GloVe": "matriz X ≈ 85 min (1 hilo) + 50 iteraciones*"},
 "Hardware": {"SGNS": "GPU GTX 1660 SUPER (6 GB)", "gensim": "CPU Ryzen 7 3700X, 8 hilos", "GloVe": "2 x Xeon E5-2658 2.1 GHz, 32 núcleos*"},
 "Analogías semánticas % (3CosAdd / 3CosMul)": {n: f"{sysres[n]['add']['sem']*100:.1f} / {sysres[n]['mul']['sem']*100:.1f}" for n in EMB},
 "Analogías sintácticas % (3CosAdd / 3CosMul)": {n: f"{sysres[n]['add']['syn']*100:.1f} / {sysres[n]['mul']['syn']*100:.1f}" for n in EMB},
 "Analogías total % (3CosAdd / 3CosMul)": {n: f"{sysres[n]['add']['total']*100:.1f} / {sysres[n]['mul']['total']*100:.1f}" for n in EMB},
 "Spearman WordSim-353": {n: f"{ws_sims[n]['spearman']:.3f}" for n in EMB},
 "Spearman SimLex-999": {n: f"{sl_sims[n]['spearman']:.3f}" for n in EMB},
 "Coseno medio entre vectores diferencia (paralelismo)": {n: f"{par[n]['_promedio']:.3f}" for n in EMB},
 "Tokens de AG News fuera del vocabulario %": {n: f"{oov_df.loc[n, 'tokens AG fuera de vocab %']:.2f}" for n in EMB},
 "F1-macro en test de AG News (variante)": {n: f"{cls[(n, 1.0)]['test']['f1']:.4f} ({cls[(n, 1.0)]['test']['variant']})" for n in EMB},
}
cmp_df = pd.DataFrame(cmp_rows).T[["SGNS", "gensim", "GloVe"]]; pd.set_option("display.max_colwidth", 80); display(cmp_df)
print("* GloVe: tiempos reportados por Pennington et al. (2014) para el corpus de 6 000 M de tokens: ≈85 min para construir X con 1 hilo, 14 min/iteración con 300 dim y 32 núcleos; usan 50 iteraciones para dim < 300 (el tiempo por iteración a 100 dim no se reporta).")
print("Evaluación de analogías: vocabulario compartido de 30 000 palabras, %d preguntas (%.1f %% del total)." % (cov["n_eval"], cov["n_eval"] / cov["n_total"] * 100))

(INF / "tabla_comparativa.tex").write_text(tex_table(["Métrica", "SGNS (propio)", "Word2Vec gensim", "GloVe-100"], [[k, *v] for k, v in cmp_df.iterrows()], "p{5.6cm}lll"), encoding="utf-8")
""")

code(r"""
# ---- Tablas auxiliares en LaTeX para el informe (iteraciones y analogías por categoría) ----
it_rows = [[n, f"{r['tokens corpus']/1e6:.1f}", int(r["d"]), int(r["W"]), int(r["K"]), f"{r['t']:g}", f"{r['pérdida final']:.3f}", f"{r['ana sem %']:.1f}", f"{r['ana syn %']:.1f}", f"{r['ana total %']:.1f}", f"{r['WS353 ρ']:.3f}",
            f"{r['t/epoch (s)']:.0f}", f"{r['t total (min)']:.1f}", f"{r['GPU pico (MB)']:.0f}"] for n, r in runs_df.iterrows()]
(INF / "tabla_iteraciones.tex").write_text(tex_table(["Corrida", "Tokens (M)", "d", "W", "K", "t", "Pérdida", "Sem.", "Sin.", "Total", "WS353", "s/ep", "min", "MB GPU"], it_rows, "lrrrrrrrrrrrrr"), encoding="utf-8")
cat_rows = [[c.replace("gram", "g").replace("-", " ") if c in CATS else c, *[f"{sys_df.loc[c, f'{n} {m}']:.1f}" for n in EMB for m in ("3CosAdd", "3CosMul")]] for c in sys_df.index]
(INF / "tabla_categorias.tex").write_text(tex_table(["Categoría", "SGNS add", "SGNS mul", "gensim add", "gensim mul", "GloVe add", "GloVe mul"], cat_rows, "lrrrrrr"), encoding="utf-8")

# ---- Vectores del mejor modelo (formato gensim .kv; se carga con KeyedVectors.load) ----
kv_best = KeyedVectors(sgns_E.shape[1]); kv_best.add_vectors(words, sgns_E); kv_best.save(str(VEC / "sgns_best_d100.kv"))
print("vectores guardados:", VEC / "sgns_best_d100.kv", f"({(VEC / 'sgns_best_d100.kv').stat().st_size / 2**20:.0f} MB) | prueba:", KeyedVectors.load(str(VEC / "sgns_best_d100.kv")).most_similar("king", topn=3))
""")
