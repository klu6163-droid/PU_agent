"""Generate human review report."""

from __future__ import annotations

from pathlib import Path

from .schemas import ExtractionResult


def generate_report(output_dir: Path, result: ExtractionResult):
    """Generate a markdown report for human review."""
    report_path = output_dir / "extraction_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction Report - {result.folder_name}\n\n")

        lit = result.literature
        f.write("## Literature\n\n")
        f.write(f"- **Title:** {lit.title}\n")
        f.write(f"- **Authors:** {', '.join(lit.authors)}\n")
        f.write(f"- **Journal:** {lit.journal} ({lit.year})\n")
        f.write(f"- **DOI:** {lit.doi}\n")
        f.write(f"- **PDF:** {lit.pdf_filename}\n\n")

        f.write("## Samples\n\n")
        if result.samples:
            for s in result.samples:
                f.write(f"### {s.sample_id}\n")
                f.write(f"- Soft segment: {s.soft_segment} (Mn: {s.soft_segment_mn})\n")
                f.write(f"- Diisocyanate: {s.diisocyanate}\n")
                f.write(f"- Chain extender: {s.chain_extender}\n")
                f.write(f"- Hard segment content: {s.hard_segment_content}\n")
                f.write(f"- NCO/OH ratio: {s.nco_oh_ratio}\n\n")
        else:
            f.write("No sample metadata was extracted in this run.\n\n")

        f.write("## Mechanical Properties\n\n")
        if result.mechanical:
            for m in result.mechanical:
                missing = []
                for field in ["young_modulus", "tensile_strength", "elongation_at_break", "toughness"]:
                    if getattr(m, field) is None:
                        missing.append(field)
                status = "COMPLETE" if not missing else f"missing: {', '.join(missing)}"
                f.write(f"- **{m.sample_id}:** {status}\n")
                f.write(f"  - Young's modulus: {m.young_modulus} MPa\n")
                f.write(f"  - Tensile strength: {m.tensile_strength} MPa\n")
                f.write(f"  - Elongation at break: {m.elongation_at_break}%\n")
                f.write(f"  - Toughness: {m.toughness} MJ/m^3\n\n")
        else:
            f.write("No mechanical properties were extracted in this run.\n\n")

        f.write("## Extracted Curves\n\n")
        if result.curves:
            curve_types = {}
            for c in result.curves:
                curve_types.setdefault(c.curve_type, []).append(c)
            for ctype, clist in curve_types.items():
                f.write(f"### {ctype.upper()} ({len(clist)} curves)\n")
                for c in clist:
                    f.write(f"- {c.label or c.sample_id}: {len(c.x)} points, confidence={c.confidence}\n")
                f.write("\n")
        else:
            f.write("No curves were extracted in this run.\n\n")

        if result.warnings:
            f.write("## Warnings\n\n")
            for w in result.warnings:
                f.write(f"- {w}\n")

        f.write("\n## Evidence Summary\n\n")
        source_counts = {}
        for e in result.evidence:
            source_counts[e.source_type] = source_counts.get(e.source_type, 0) + 1
        if source_counts:
            for stype, count in source_counts.items():
                f.write(f"- {stype}: {count} fields\n")
        else:
            f.write("No evidence records were produced in this run.\n")
