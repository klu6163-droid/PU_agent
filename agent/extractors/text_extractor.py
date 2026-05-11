"""PDF text and table extraction — wraps utils_pdf."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseExtractor, ExtractionContext
from ..output.schemas import LiteratureInfo

logger = logging.getLogger(__name__)


def _read_molecule_names(folder: Path) -> list[str]:
    """Read sample names from molecule.txt."""
    mol_file = folder / "molecule.txt"
    if not mol_file.exists():
        return []
    names = []
    for line in mol_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            names.append(line)
    return names


def _find_pdfs(folder: Path) -> tuple[Path | None, Path | None]:
    """Find main and SI PDFs in a folder."""
    from ..utils_pdf import find_pdfs
    result = find_pdfs(folder)
    return result.get("main"), result.get("si")


def prepare_context(folder: Path, config) -> ExtractionContext:
    """Prepare extraction context for a folder: read text, tables, molecule names."""
    from ..utils_pdf import (
        extract_text_pymupdf, extract_text_pdfplumber,
        extract_tables_pdfplumber, classify_pdf,
    )
    from ..extractors.figure_detector import build_figure_index

    ctx = ExtractionContext(folder=folder, folder_name=folder.name)
    ctx.sample_names = _read_molecule_names(folder)

    main_pdf, si_pdf = _find_pdfs(folder)
    ctx.main_pdf = main_pdf
    ctx.si_pdf = si_pdf

    # Extract text
    if main_pdf:
        ctx.main_text = extract_text_pymupdf(main_pdf)
        logger.info(f"  Main text: {len(ctx.main_text)} chars from {main_pdf.name}")
    if si_pdf:
        ctx.si_text = extract_text_pymupdf(si_pdf)
        logger.info(f"  SI text: {len(ctx.si_text)} chars from {si_pdf.name}")

    # Extract tables
    if main_pdf:
        ctx.main_tables = extract_tables_pdfplumber(main_pdf)
        logger.info(f"  Main tables: {len(ctx.main_tables)}")
    if si_pdf:
        ctx.si_tables = extract_tables_pdfplumber(si_pdf)
        logger.info(f"  SI tables: {len(ctx.si_tables)}")

    # Build figure index
    ctx.figure_index = build_figure_index(ctx)
    logger.info(f"  Figures: {len(ctx.figure_index)}")

    return ctx
