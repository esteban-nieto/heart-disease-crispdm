"""
Convierte los scripts de fase en notebooks .ipynb ejecutados.

- Marcadores "# %% [markdown]" -> celda Markdown (líneas "# ..." sin el prefijo).
- Marcadores "# %%"            -> celda de código.
- Encabezados "# ---------- Título ----------" -> celda Markdown con el título.

Uso (desde la raíz del proyecto):
    python notebooks/_convertir_a_ipynb.py
"""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parent
RAIZ = NOTEBOOKS.parent
FASES = [
    "01_comprension_y_eda",
    "02_preparacion_y_base_de_datos",
    "03_modelamiento_y_evaluacion",
    "06_robustez_y_equidad",
]
SIN_EJECUTAR = {"03_modelamiento_y_evaluacion"}
SOLO = set(sys.argv[1:])  # uso: python _convertir_a_ipynb.py 06_robustez_y_equidad
RE_MD = re.compile(r"^# %% \[markdown\]\s*$")
RE_CODE = re.compile(r"^# %%\s*$")
RE_SECCION = re.compile(r"^# -{10} (.+?) -{10}\s*$")


def dividir(texto):
    celdas, modo, buf = [], "codigo", []

    def cerrar():
        if not buf:
            return
        contenido = "\n".join(buf).strip("\n")
        if modo == "md":
            lineas = [l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l)
                      for l in contenido.splitlines()]
            contenido = "\n".join(lineas).strip("\n")
            if contenido:
                celdas.append(nbformat.v4.new_markdown_cell(contenido))
        elif contenido.strip():
            celdas.append(nbformat.v4.new_code_cell(contenido))

    for linea in texto.splitlines():
        if RE_MD.match(linea):
            cerrar(); buf = []; modo = "md"
        elif RE_CODE.match(linea):
            cerrar(); buf = []; modo = "codigo"
        elif RE_SECCION.match(linea):
            cerrar(); buf = []; modo = "codigo"
            titulo = RE_SECCION.match(linea).group(1)
            celdas.append(nbformat.v4.new_markdown_cell(f"## {titulo}"))
        else:
            buf.append(linea)
    cerrar()
    return celdas


def main():
    for fase in FASES:
        if SOLO and fase not in SOLO:
            continue
        origen = NOTEBOOKS / f"{fase}.py"
        texto = origen.read_text(encoding="utf-8")
        # En Jupyter no existe __file__: se usa el directorio de trabajo (notebooks/).
        texto = texto.replace("RAIZ = Path(__file__).resolve().parents[1]",
                              "RAIZ = Path.cwd().parent")
        celdas = dividir(texto)
        nb = nbformat.v4.new_notebook(cells=celdas)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        cliente = NotebookClient(nb, timeout=900, kernel_name="python3",
                                 resources={"metadata": {"path": str(NOTEBOOKS)}})
        if fase in SIN_EJECUTAR:
            # La fase 3 tarda ~30 min (SHAP). Su salida ya está en reports/logs/;
            # el notebook se guarda sin ejecutar para no repetir ese cálculo.
            print(f"Guardando {fase}.ipynb sin ejecutar (salida en reports/logs/)")
            nbformat.write(nb, NOTEBOOKS / f"{fase}.ipynb")
            continue
        print(f"Ejecutando {fase}.ipynb ({len(celdas)} celdas)...")
        cliente.execute()
        salida = NOTEBOOKS / f"{fase}.ipynb"
        nbformat.write(nb, salida)
        print("  guardado:", salida)


if __name__ == "__main__":
    sys.exit(main())
