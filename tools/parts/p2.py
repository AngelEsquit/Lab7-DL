CELLS = []
def md(t): CELLS.append(("md", t))
def code(t): CELLS.append(("code", t))

md(r"""
## 3. Investigación: embeddings en PyTorch y gensim
La investigación completa (`nn.Embedding`, `from_pretrained`, `nn.EmbeddingBag`, CBOW vs. skip-gram, las dos matrices de Word2Vec y Word2Vec vs. GloVe) está redactada en el **informe** (`informe/informe.tex`, sección 2). Aquí solo se deja una demostración ejecutable de `nn.EmbeddingBag` con `offsets` y sus tres modos de agregación.
""")

code(r"""
# Demostración de EmbeddingBag con offsets: 3 documentos de longitudes 3, 2 y 1
bag_vals = torch.arange(12, dtype=torch.float).view(6, 2)       # filas [0,1],[2,3],...,[10,11]
flat    = torch.tensor([0, 1, 2,   3, 4,   5])                   # IDs de los 3 documentos concatenados
offsets = torch.tensor([0, 3, 5])                                # posición de inicio de cada documento
for mode in ("mean", "sum", "max"):
    bag = nn.EmbeddingBag.from_pretrained(bag_vals, mode=mode, freeze=True)
    print(mode, "\n", bag(flat, offsets))
print("comprobación manual del 'mean' del primer documento:", bag_vals[[0, 1, 2]].mean(0))
""")
