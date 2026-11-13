"""
Post-proceso de reportes: CSV para Excel y documentos Word.

1. CSV: se reescriben en formato que Excel en configuración española abre en columnas:
   - separador ";" y decimal "," (estándar de Excel en español),
   - UTF-8 con BOM (para tildes y ñ),
   - las celdas tipo diccionario (p. ej. {0.0: 0.047, 1.0: 0.170}) se convierten en filas:
     una fila por nivel, con columnas nivel, tasa_evento y p_valor.
2. Word: los informes Markdown se convierten a .docx con encabezados, tablas,
   listas, figuras y pantallazos incrustados.

Ejecutar desde la raíz del proyecto:
    python notebooks/05_formato_excel_y_word.py
"""
import ast
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SEP = ";"
DECIMAL = ","
COLUMNAS_DICCIONARIO = {"tasa_evento_por_nivel", "tasa_riesgo_por_nivel",
                        "tasa_evento", "tasa_riesgo"}


def _es_diccionario(serie):
    """Una columna es diccionario si sus valores tienen forma de dict: {...}."""
    muestra = serie.dropna().astype(str)
    return len(muestra) > 0 and muestra.str.startswith("{").all()


def a_tabular(df):
    """Expande columnas tipo diccionario en filas (una por nivel)."""
    col = next((c for c in df.columns if _es_diccionario(df[c])), None)
    if col is None:
        return df
    filas = []
    for _, fila in df.iterrows():
        base = {k: v for k, v in fila.items() if k != col}
        dic = ast.literal_eval(str(fila[col]))
        for nivel, tasa in dic.items():
            nivel_txt = str(int(nivel)) if isinstance(nivel, float) and nivel.is_integer() \
                else str(nivel)
            filas.append({"variable": base.get("variable"),
                          "nivel": nivel_txt,
                          "tasa_evento": round(float(tasa), 4),
                          "p_valor": base.get("p_valor")})
    return pd.DataFrame(filas)


def leer_csv(ruta):
    """Lee CSV en cualquiera de los dos formatos (coma/punto o ;/coma)."""
    cabecera = Path(ruta).read_text(encoding="utf-8-sig").splitlines()[0]
    if ";" in cabecera:
        return pd.read_csv(ruta, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(ruta, encoding="utf-8-sig")


def reescribir_csv(ruta):
    df = leer_csv(ruta)
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "variable"})
    if "variable" not in df.columns and df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "variable"})
    df = a_tabular(df)
    df.to_csv(ruta, sep=SEP, decimal=DECIMAL, index=False,
              encoding="utf-8-sig", float_format="%.6g")
    return df


# ---------------- Word ----------------

def _fondo_celda(celda, color_hex):
    props = celda._tc.get_or_add_tcPr()
    sombreado = OxmlElement("w:shd")
    sombreado.set(qn("w:val"), "clear")
    sombreado.set(qn("w:color"), "auto")
    sombreado.set(qn("w:fill"), color_hex)
    props.append(sombreado)


def _runs_inline(parrafo, texto, tamano=None):
    """Aplica **negrita**, `código` y quita enlaces [texto](url)."""
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)
    partes = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", texto)
    for parte in partes:
        if not parte:
            continue
        if parte.startswith("**") and parte.endswith("**"):
            r = parrafo.add_run(parte[2:-2]); r.bold = True
        elif parte.startswith("`") and parte.endswith("`"):
            r = parrafo.add_run(parte[1:-1]); r.font.name = "Consolas"
        else:
            r = parrafo.add_run(parte)
        if tamano:
            r.font.size = Pt(tamano)


def _tabla(doc, filas):
    filas = [f for f in filas if not all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in f)]
    ncol = max(len(f) for f in filas)
    t = doc.add_table(rows=len(filas), cols=ncol)
    t.style = "Table Grid"
    for i, fila in enumerate(filas):
        for j in range(ncol):
            texto = fila[j].strip() if j < len(fila) else ""
            celda = t.cell(i, j)
            celda.text = ""
            p = celda.paragraphs[0]
            _runs_inline(p, texto, tamano=8.5)
            if i == 0:
                for r in p.runs:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                _fondo_celda(celda, "2A7AB0")
    doc.add_paragraph()


def markdown_a_word(md_path, salida, ancho_img_cm=15):
    md_path = Path(md_path)
    base = md_path.parent
    doc = Document()
    estilo = doc.styles["Normal"]
    estilo.font.name = "Calibri"
    estilo.font.size = Pt(10.5)
    for seccion in doc.sections:
        seccion.left_margin = seccion.right_margin = Cm(2.2)
        seccion.top_margin = seccion.bottom_margin = Cm(2)

    lineas = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    en_codigo = False
    bloque_codigo = []
    while i < len(lineas):
        linea = lineas[i]
        if linea.strip().startswith("```"):
            if en_codigo:
                p = doc.add_paragraph()
                r = p.add_run("\n".join(bloque_codigo)); r.font.name = "Consolas"
                r.font.size = Pt(8.5)
                bloque_codigo = []
            en_codigo = not en_codigo
            i += 1
            continue
        if en_codigo:
            bloque_codigo.append(linea)
            i += 1
            continue
        s = linea.strip()
        if not s or s == "---":
            i += 1
            continue
        if s.startswith("|"):
            filas = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                filas.append(lineas[i].strip().strip("|").split("|"))
                i += 1
            _tabla(doc, filas)
            continue
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            ruta = (base / m.group(2)).resolve()
            if ruta.exists():
                doc.add_picture(str(ruta), width=Cm(ancho_img_cm))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if m.group(1):
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    rc = cap.add_run(m.group(1)); rc.italic = True; rc.font.size = Pt(8.5)
            else:
                doc.add_paragraph(f"[Imagen no encontrada: {m.group(2)}]")
            i += 1
            continue
        if s.startswith("### "):
            doc.add_heading(s[4:], level=3)
        elif s.startswith("## "):
            doc.add_heading(s[3:], level=2)
        elif s.startswith("# "):
            doc.add_heading(s[2:], level=1)
        elif re.match(r"^[-*] ", s):
            p = doc.add_paragraph(style="List Bullet")
            _runs_inline(p, s[2:])
        elif re.match(r"^\d+\. ", s):
            p = doc.add_paragraph(style="List Number")
            _runs_inline(p, re.sub(r"^\d+\. ", "", s))
        else:
            # Línea de párrafo: une líneas consecutivas de texto normal.
            bloque = [s]
            while i + 1 < len(lineas):
                sig = lineas[i + 1].strip()
                if (not sig or sig.startswith(("#", "|", "!", "```", "- ", "* ", "---"))
                        or re.match(r"^\d+\. ", sig)):
                    break
                bloque.append(sig)
                i += 1
            p = doc.add_paragraph()
            _runs_inline(p, " ".join(bloque))
        i += 1

    salida = Path(salida)
    doc.save(salida)
    return salida


def main(raiz):
    raiz = Path(raiz)
    reportes = raiz / "reports"
    # CSV
    for csv in sorted(reportes.glob("*.csv")):
        df = reescribir_csv(csv)
        print(f"CSV reescrito: {csv.name}  ({df.shape[0]} filas x {df.shape[1]} columnas)")
    # Word
    for md in sorted(reportes.glob("*.md")):
        if md.name.lower() in ("readme.md",):
            continue
        salida = reportes / (md.stem + ".docx")
        markdown_a_word(md, salida)
        print(f"Word generado: {salida}")


if __name__ == "__main__":
    for ruta in sys.argv[1:] or ["."]:
        main(ruta)
