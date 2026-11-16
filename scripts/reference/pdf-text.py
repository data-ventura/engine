"""Extracts the text of dv-render's golden PDFs with pdfium, for the PDF export's text test.

Usage: pdf-text.py <golden dir> <out.json>

Run with: uv run --with pypdfium2 python scripts/reference/pdf-text.py crates/dv-render/tests/golden fixtures/reference/pdf-text.json

For each `<name>.pdf`, records the page size in points and every character pdfium extracts, with
its box (left, bottom, right, top) in points, y up. pdfium is the PDF engine in Chromium and is
independent of the writer, so the test checks that a common reader finds each label as text where
the chart drew it.
"""

import json
import pathlib
import sys

import pypdfium2 as pdfium

out = {"pdfium": pdfium.PDFIUM_INFO.version}
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.pdf")):
    pdf = pdfium.PdfDocument(path)
    page = pdf[0]
    width, height = page.get_size()
    text = page.get_textpage()
    chars = []
    for i in range(text.count_chars()):
        ch = text.get_text_range(i, 1)
        box = text.get_charbox(i, loose=False)
        chars.append({"char": ch, "box": [round(v, 3) for v in box]})
    out[path.stem] = {"width": width, "height": height, "chars": chars}
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
