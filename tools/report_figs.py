# Figuras compactas para el informe (lee artifacts/runs/*.json; se puede ejecutar desde la raíz del repo).
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART, FIG = Path("artifacts/runs"), Path("figures")
FIG.mkdir(exist_ok=True)
names = ["d50", "d100_base", "d300", "frac25", "frac50", "W2", "W10", "K15", "t1e-5"]
runs = {n: json.load(open(ART / f"{n}.json")) for n in names}
style = {"d50": ("tab:blue", ":"), "d100_base": ("k", "-"), "d300": ("tab:blue", "--"), "frac25": ("tab:orange", ":"), "frac50": ("tab:orange", "--"),
         "W2": ("tab:green", ":"), "W10": ("tab:green", "--"), "K15": ("tab:red", "-"), "t1e-5": ("tab:purple", "-")}

fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.7))
for n in names:
    h = runs[n]["history"]; ep = [x["epoch"] for x in h]; c, ls = style[n]
    kw = dict(color=c, ls=ls, lw=2.2 if n == "d100_base" else 1.3, label=n)
    if n != "K15":
        ax[0].plot(ep, [x["loss"] for x in h], **kw)      # K15 tiene otra escala de pérdida (más términos)
    ax[1].plot(ep, [x["ana_total"] * 100 for x in h], **kw)
    ax[2].plot(ep, [x["ws353"] for x in h], **kw)
ax[0].set_ylabel("pérdida (sin K15)"); ax[1].set_ylabel("analogías total (%)"); ax[2].set_ylabel("Spearman WS-353")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=.3); a.tick_params(labelsize=8)
ax[2].legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
plt.tight_layout(); plt.savefig(FIG / "curvas_compacto.png", dpi=170, bbox_inches="tight")
print("ok", FIG / "curvas_compacto.png")
