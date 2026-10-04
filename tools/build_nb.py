import glob, importlib.util, os, sys
import nbformat as nbf
S = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
nb = nbf.v4.new_notebook()
cells = []
for f in sorted(glob.glob(os.path.join(S, "parts", "p*.py"))):
    spec = importlib.util.spec_from_file_location("part", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for kind, text in m.CELLS:
        text = text.strip("\n")
        cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, OUT)
print("cells:", len(cells))
