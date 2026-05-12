"""Write extraction results to CSV and JSON files."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from .schemas import ExtractionResult

logger = logging.getLogger(__name__)


def write_results(output_dir: Path, result: ExtractionResult):
    """Write all extraction results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(output_dir)

    _write_json(output_dir / "literature.json", result.literature.model_dump())

    if result.samples:
        _write_csv(output_dir / "metadata.csv", [s.model_dump() for s in result.samples])

    if result.mechanical:
        _write_csv(output_dir / "mechanical.csv", [m.model_dump() for m in result.mechanical])

    curves_dir = output_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    for curve in result.curves:
        safe_name = _curve_filename(curve)
        _write_curve_csv(curves_dir / f"{safe_name}.csv", curve)

    if result.evidence:
        _write_csv(output_dir / "evidence.csv", [e.model_dump() for e in result.evidence])

    if result.warnings:
        with open(output_dir / "warnings.md", "w", encoding="utf-8") as f:
            f.write(f"# Warnings - {result.folder_name}\n\n")
            for w in result.warnings:
                f.write(f"- {w}\n")

    _write_json(output_dir / "extraction_result.json", result.model_dump())
    logger.info(f"  Output written to {output_dir}")


def _write_csv(path: Path, rows: list[dict]):
    """Write list of dicts to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data):
    """Write data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _write_curve_csv(path: Path, curve):
    """Write curve data to CSV with per-point provenance."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x",
            "y",
            "source_pdf",
            "source_page",
            "source_figure",
            "caption",
            "extraction_method",
            "confidence",
            "x_scale",
            "y_scale",
        ])
        for x, y in zip(curve.x, curve.y):
            writer.writerow([
                x,
                y,
                curve.source_pdf,
                curve.source_page,
                curve.source_figure,
                curve.caption,
                curve.extraction_method,
                curve.confidence,
                curve.x_scale,
                curve.y_scale,
            ])


def _safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:100]


def _curve_filename(curve) -> str:
    """Build a unique, stable filename for one extracted curve."""
    parts = [
        curve.sample_id,
        curve.curve_type,
        curve.source_figure,
        f"p{curve.source_page}" if curve.source_page else "",
        curve.label,
    ]
    return _safe_filename("_".join(str(part) for part in parts if part))


def _remove_stale_outputs(output_dir: Path):
    """Remove generated files from a previous run before writing this run."""
    for filename in (
        "literature.json",
        "metadata.csv",
        "mechanical.csv",
        "evidence.csv",
        "warnings.md",
        "extraction_result.json",
        "extraction_report.md",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    curves_dir = output_dir / "curves"
    if curves_dir.exists():
        for curve_file in curves_dir.glob("*.csv"):
            curve_file.unlink()
