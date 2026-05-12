"""Figure detection and classification from extracted text."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

PLOT_TYPE_KEYWORDS = {
    "stress_strain": [
        "stress", "strain", "tensile", "engineering stress", "true stress",
        "elongation", "stress-strain", "loading", "unloading",
        "toughness", "fracture", "modulus", "mpa", "strength",
        "stretching", "tensile test", "mechanical properties",
        "elongation at break", "young's modulus", "yield",
    ],
    "ftir": [
        "ftir", "infrared", "ir spectrum", "wavenumber", "transmittance",
        "absorbance", "ft-ir", "fourier transform infrared",
        "ir spectra", "ftir spectra", "cm-1", "cm⁻¹",
        "hydrogen bond", "h-bond", "carbonyl", "nh stretching",
    ],
    "saxs": [
        "saxs", "small angle", "small-angle", "scattering intensity",
        "nm^-1", "nm-1", "q (nm", "x-ray scattering",
        "small-angle x-ray", "scattering curve", "scattering pattern",
        "phase separation", "domain size", "long period",
    ],
    "waxs": [
        "waxs", "wide angle", "wide-angle", "2theta", "2 theta",
        "xrd", "diffraction", "crystallinity", "crystallite",
        "wide-angle x-ray", "powder diffraction",
    ],
    "dsc": [
        "dsc", "differential scanning", "heat flow", "thermal analysis",
        "glass transition", "melting", "crystallization", "dma",
        "dynamic mechanical", "storage modulus", "loss modulus", "tan delta",
        "enthalpy", "thermogram", "thermal transition",
        "dsc curve", "dsc analysis",
    ],
    "tem": ["tem ", "tem image", "transmission electron", "microscopy", "morphology"],
    "afm": ["afm", "atomic force", "phase image"],
}


def detect_plot_types(caption: str) -> list[str]:
    """Classify a figure caption into plot types."""
    caption_lower = caption.lower()
    detected = []
    for ptype, keywords in PLOT_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in caption_lower:
                detected.append(ptype)
                break
    if re.search(r"\bT[gmc]\b", caption):
        detected.append("dsc")
    return detected if detected else ["unknown"]


def parse_figure_caption(caption_text: str) -> dict | None:
    """Parse a figure caption to extract figure ID, panel, and text."""
    pattern = r'(?:Figure|Fig\.?)\s*(S?\d+)\s*([a-z])?\s*[:\.]?\s*(.*)'
    m = re.search(pattern, caption_text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    figure_id = f"Figure {m.group(1)}"
    panel_id = m.group(2) or ""
    caption = m.group(3).strip()[:500]

    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "caption": caption,
        "plot_types": detect_plot_types(caption),
    }


def _is_toc_page(text: str, pos: int) -> bool:
    """Check whether text around pos looks like a table-of-contents listing."""
    start = max(0, pos - 500)
    end = min(len(text), pos + 1500)
    snippet = text[start:end]
    fig_refs = re.findall(r'Figure\s+S?\d+', snippet, re.IGNORECASE)
    return len(fig_refs) >= 5


def build_figure_index(context) -> list[dict]:
    """Build figure index from extracted text of main and SI PDFs."""
    figures = []
    seen = set()

    for text, pdf_path, pdf_type in [
        (context.main_text, context.main_pdf, "main"),
        (context.si_text, context.si_pdf, "si"),
    ]:
        if not text or not pdf_path:
            continue

        pattern = r'((?:Figure|Fig\.?)\s*S?\d+\s*[a-z]?\s*[:\.]?.{0,1000})'
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)

        for match in matches:
            parsed = parse_figure_caption(match.group(1).strip())
            if not parsed:
                continue
            if _is_toc_page(text, match.start()):
                continue

            key = f"{parsed['figure_id']}_{parsed['panel_id']}_{pdf_type}"
            if key in seen:
                continue
            seen.add(key)

            page_markers = list(re.finditer(r'--- PAGE (\d+) ---', text[:match.start()]))
            page_num = int(page_markers[-1].group(1)) if page_markers else 0

            figures.append({
                "figure_id": parsed["figure_id"],
                "panel_id": parsed["panel_id"],
                "caption": parsed["caption"],
                "plot_types": parsed["plot_types"],
                "pdf_name": pdf_path.name,
                "pdf_type": pdf_type,
                "page_number": page_num,
                "sample_labels": _extract_sample_labels(parsed["caption"], context.sample_names),
            })

    return figures


def _extract_sample_labels(caption: str, sample_names: list[str]) -> list[str]:
    """Find sample names that appear in a figure caption."""
    return [name for name in sample_names if name in caption]
