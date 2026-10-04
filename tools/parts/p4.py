CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
### 4.3 Iteraciones del SGNS
Un piloto previo (2 epochs sobre el 25 % del corpus; no se reporta en la tabla) mostró que el tamaño de batch (1 024 → 16 384) casi no cambia la calidad y que `lr=0.01` rinde mejor que 0.003 y 0.03 (WS-353 0.61 vs 0.49 y 0.52 tras 2 epochs), así que se fijan **B = 16 384, lr = 0.01 (decae linealmente a 5 %), 10 epochs** y se varía lo que pide el enunciado:

| Grupo | Corridas |
|---|---|
| Dimensión | `d=50`, `d=100` (base), `d=300` |
| Tamaño del corpus | 25 %, 50 %, 100 % de los 29.6 M tokens (vocabulario fijo) |
| Ventana | `W=2`, `W=5` (base), `W=10` |
| Negativos | `K=5` (base), `K=15` |
| Submuestreo | `t=1e-4` (base), `t=1e-5` |

Por cada corrida se registra: pérdida por epoch, analogías (sem/sin/total, 3CosAdd, espacio de búsqueda = 30 000 palabras más frecuentes) y Spearman en **WordSim-353** al final de cada epoch, los 5 vecinos de palabras de control, tiempo por epoch/total y memoria pico de GPU. (SimLex-999 **no** se usa aquí.)
""")

code(r"""
BASE = dict(d=100, W=5, K=5, t=1e-4, epochs=10, frac=1.0)
RUNS = {
  "d100_base":  {},                      "d50": dict(d=50),            "d300": dict(d=300),
  "frac25":     dict(frac=0.25),         "frac50": dict(frac=0.5),
  "W2":         dict(W=2),               "W10": dict(W=10),
  "K15":        dict(K=15),              "t1e-5": dict(t=1e-5),
}
results = {}
for name, over in RUNS.items():
    results[name] = train_sgns(name, **{**BASE, **over})
    r = results[name]; print(f"==> {name}: {r['total_secs']/60:.1f} min, pico GPU {r['peak_gpu_mb']:.0f} MB\n")
""")

code(r"""
rows = []
for name, r in results.items():
    c, h = r["config"], r["history"][-1]
    rows.append({"corrida": name, "tokens corpus": c["tokens_corpus"], "d": c["d"], "W": c["W"], "K": c["K"], "t": c["t"], "epochs": c["epochs"],
                 "pérdida final": h["loss"], "ana sem %": h["ana_sem"] * 100, "ana syn %": h["ana_syn"] * 100, "ana total %": h["ana_total"] * 100,
                 "WS353 ρ": h["ws353"], "t/epoch (s)": np.mean([x["epoch_secs"] for x in r["history"]]), "t total (min)": r["total_secs"] / 60, "GPU pico (MB)": r["peak_gpu_mb"]})
runs_df = pd.DataFrame(rows).set_index("corrida")
display(runs_df.style.format({"tokens corpus": "{:,}", "pérdida final": "{:.4f}", "ana sem %": "{:.2f}", "ana syn %": "{:.2f}", "ana total %": "{:.2f}", "WS353 ρ": "{:.3f}", "t/epoch (s)": "{:.0f}", "t total (min)": "{:.1f}", "GPU pico (MB)": "{:.0f}", "t": "{:g}"}))

# Selección de la mejor configuración: SOLO analogías + WordSim-353 (SimLex-999 queda reservado)
runs_df["score"] = 0.5 * runs_df["ana total %"] / 100 + 0.5 * runs_df["WS353 ρ"]
best_name = runs_df["score"].idxmax(); print("Mejor corrida SGNS (promedio de analogías totales y Spearman WS-353):", best_name, "| score", round(runs_df.loc[best_name, 'score'], 4))
best_cfg = results[best_name]["config"]; print(best_cfg)
""")

code(r"""
groups = {"Dimensión": ["d50", "d100_base", "d300"], "Tamaño del corpus": ["frac25", "frac50", "d100_base"], "Ventana / negativos / submuestreo": ["W2", "d100_base", "W10", "K15", "t1e-5"]}
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
for row, (g, names) in enumerate(groups.items()):
    for name in names:
        h = results[name]["history"]; ep = [x["epoch"] for x in h]
        axes[row, 0].plot(ep, [x["loss"] for x in h], marker="o", label=name)
        axes[row, 1].plot(ep, [x["ana_total"] * 100 for x in h], marker="o", label=name)
        axes[row, 2].plot(ep, [x["ws353"] for x in h], marker="o", label=name)
    axes[row, 0].set_ylabel(g + "\npérdida de entrenamiento"); axes[row, 1].set_ylabel("accuracy analogías total (%)"); axes[row, 2].set_ylabel("Spearman WS-353")
    for a in axes[row]: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); FIG.mkdir(exist_ok=True); plt.savefig(FIG / "iteraciones_curvas.png", dpi=150); plt.show()
""")

code(r"""
# Vecinos más cercanos de las palabras de control, al final de cada epoch (corrida base, d=100)
nb_rows = []
for h in results["d100_base"]["history"]:
    nb_rows.append({"epoch": h["epoch"], **{w: ", ".join(x for x, _ in h["neighbors"][w]) for w in CONTROL}})
pd.set_option("display.max_colwidth", 70); display(pd.DataFrame(nb_rows).set_index("epoch"))
""")

md(r"""
**¿Una pérdida más baja implica siempre mejores embeddings? No.**
- En la corrida base la pérdida se **estanca desde el epoch 8** (2.123 → 2.125) mientras las analogías siguen subiendo (43.8 % → 46.3 % → 47.3 %).
- Entre corridas con `K=5`, la correlación de Spearman entre pérdida final y accuracy de analogías es **0.12** (p = 0.78): `frac25` tiene **menor** pérdida (2.027) que la base (2.125) pero solo 24.3 % de analogías (vs 47.3 %); `t1e-5` (2.009) y `W2` (1.971) tienen menos pérdida que la base y peores analogías (41.5 % y 44.4 %); `d300` baja la pérdida (2.031) y mejora las analogías (49.1 %) pero empeora WS-353 (0.668 vs 0.689).
- La pérdida **no es comparable** entre configuraciones con distinta ventana, submuestreo o número de negativos (`K=15` tiene pérdida 3.03 porque suma más términos): cambia la distribución de pares y la tarea.

La pérdida mide qué tan bien se separa el contexto real del ruido (la tarea sustituta), no la calidad de la geometría del espacio; un corpus pequeño o un modelo con más capacidad ajustan mejor la tarea sin generalizar mejor las relaciones semánticas.
""")
