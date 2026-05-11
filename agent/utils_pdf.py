"""PDF utility functions — copied from scripts/utils_pdf.py."""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


def classify_pdf(pdf_path: Path) -> str:
    """Classify a PDF as 'main' or 'si' based on filename."""
    name = pdf_path.name.lower()
    stem = pdf_path.stem.lower()
    si_keywords = ["supp", "supporting", "supplementary", "misc_information"]
    if any(kw in name for kw in si_keywords):
        return "si"

    # Treat "si" as a standalone filename token only. A plain substring match
    # misclassifies names such as "synthesis.pdf" or "physical_properties.pdf".
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
    if "si" in tokens:
        return "si"
    return "main"


def find_pdfs(folder: Path) -> dict:
    """Find main and SI PDFs in a folder. Returns {'main': Path, 'si': Path}."""
    result = {"main": None, "si": None}
    for pdf in sorted(folder.glob("*.pdf")):
        pdf_type = classify_pdf(pdf)
        if pdf_type == "si" and result["si"] is None:
            result["si"] = pdf
        elif pdf_type == "main" and result["main"] is None:
            result["main"] = pdf
    if result["main"] is None and result["si"] is not None:
        pdfs = list(folder.glob("*.pdf"))
        if len(pdfs) == 2:
            sizes = [(p, p.stat().st_size) for p in pdfs]
            sizes.sort(key=lambda x: x[1], reverse=True)
            result["main"] = sizes[0][0]
            result["si"] = sizes[1][0]
    return result


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_parts.append(f"--- PAGE {page_num + 1} ---\n{text}")
    doc.close()
    return "\n".join(text_parts)


def extract_text_pdfplumber(pdf_path: Path) -> str:
    """Extract full text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text_parts.append(f"--- PAGE {i + 1} ---\n{text}")
    return "\n".join(text_parts)


def extract_tables_pdfplumber(pdf_path: Path) -> list:
    """Extract all tables from a PDF using pdfplumber."""
    tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for tbl in page_tables:
                if tbl and len(tbl) > 1:
                    tables.append((i + 1, tbl))
    return tables


def render_page_to_image(pdf_path: Path, page_num: int, dpi: int = 400) -> Optional[bytes]:
    """Render a specific page to PNG image bytes at given DPI."""
    doc = fitz.open(str(pdf_path))
    if page_num >= len(doc):
        doc.close()
        return None
    page = doc[page_num]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def get_page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def extract_page_text(pdf_path: Path, page_num: int) -> str:
    doc = fitz.open(str(pdf_path))
    if page_num >= len(doc):
        doc.close()
        return ""
    text = doc[page_num].get_text("text")
    doc.close()
    return text


def find_figure_pages(pdf_path: Path) -> list:
    import re
    figure_pages = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        matches = re.findall(r'(?:Figure|Fig\.?)\s*(?:S?\d+(?:[a-z])?)', text, re.IGNORECASE)
        if matches:
            figure_pages.append((page_num, list(set(matches))))
    doc.close()
    return figure_pages


def get_pdf_metadata(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    meta = doc.metadata or {}
    meta["page_count"] = len(doc)
    meta["file_size"] = pdf_path.stat().st_size
    meta["filename"] = pdf_path.name
    doc.close()
    return meta
