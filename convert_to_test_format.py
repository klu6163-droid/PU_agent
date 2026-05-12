"""Convert agent_output to test format for each PU folder."""

import csv
import json
import os
import re
import sys
from pathlib import Path

PU_ROOT = Path("D:/CC Code/PU")
FOLDERS = [f"PU_{i:03d}" for i in range(1, 10)]


def safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:100]


def determine_curve_type(filename: str, curve_data: dict) -> str:
    """Determine the curve type from filename or data."""
    fn_lower = filename.lower()
    ct = curve_data.get("curve_type", "").lower()
    if "stress_strain" in fn_lower or ct == "stress_strain":
        return "stress_strain"
    if "ftir" in fn_lower or ct == "ftir":
        return "ftir"
    if "saxs" in fn_lower or ct == "saxs":
        return "saxs"
    if "dsc" in fn_lower or ct == "dsc":
        return "dsc"
    if "xrd" in fn_lower or "waxs" in fn_lower or ct in ("xrd", "waxs"):
        return "xrd"
    # Try from curve_type in JSON
    if ct in ("stress_strain", "ftir", "saxs", "dsc", "xrd", "waxs"):
        return ct
    return "unknown"


def parse_agent_curve_filename(filename: str) -> tuple[str, str]:
    """Parse sample_id and curve_type from an agent curve CSV filename."""
    name = filename.replace(".csv", "")
    for ct in ["stress_strain", "ftir", "saxs", "dsc", "xrd", "waxs"]:
        marker = f"_{ct}_"
        idx = name.lower().find(marker)
        if idx > 0:
            return name[:idx], ct
    return "", ""


def get_column_names(curve_type: str) -> tuple[str, str]:
    """Get the x,y column names for a curve type."""
    if curve_type == "stress_strain":
        return "strain", "stress"
    elif curve_type == "ftir":
        return "wavenumber", "absorbance"
    elif curve_type == "saxs":
        return "q", "intensity"
    elif curve_type == "dsc":
        return "temperature", "heat_flow"
    elif curve_type == "xrd":
        return "two_theta", "intensity"
    return "x", "y"


def process_folder(folder_name: str):
    """Convert agent_output to test format for one PU folder."""
    folder = PU_ROOT / folder_name
    agent_out = folder / "agent_output"

    if not agent_out.exists():
        print(f"  Skipping {folder_name}: no agent_output")
        return

    # Read extraction_result.json for full curve data
    result_json = agent_out / "extraction_result.json"
    if not result_json.exists():
        print(f"  Skipping {folder_name}: no extraction_result.json")
        return

    with open(result_json, encoding="utf-8") as f:
        data = json.load(f)

    sample_ids = [s["sample_id"] for s in data.get("samples", [])]
    curves = data.get("curves", [])

    # --- Copy and enhance metadata.csv ---
    src_meta = agent_out / "metadata.csv"
    if src_meta.exists():
        with open(src_meta, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            meta_rows = list(reader)

        # Add source and reference columns
        for row in meta_rows:
            row["source"] = folder_name
            row["reference"] = data.get("literature", {}).get("title", "")

        meta_path = folder / "metadata.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            if meta_rows:
                writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
                writer.writeheader()
                writer.writerows(meta_rows)
        print(f"  metadata.csv: {len(meta_rows)} samples")

    # --- Copy mechanical.csv ---
    src_mech = agent_out / "mechanical.csv"
    if src_mech.exists():
        mech_content = src_mech.read_text(encoding="utf-8")
        (folder / "mechanical.csv").write_text(mech_content, encoding="utf-8")
        mech_rows = list(csv.DictReader(mech_content.splitlines()))
        print(f"  mechanical.csv: {len(mech_rows)} samples")

    # --- Process curves ---
    # Group curves by type
    VALID_TYPES = {"stress_strain", "ftir", "saxs", "dsc", "xrd", "waxs"}
    curve_groups: dict[str, list[dict]] = {}
    for curve in curves:
        ct = curve.get("curve_type", "unknown")
        if ct not in VALID_TYPES:
            continue
        if ct not in curve_groups:
            curve_groups[ct] = []
        curve_groups[ct].append(curve)

    # Also process from CSV files in curves/ directory (which may have more detail)
    curves_dir = agent_out / "curves"
    if curves_dir.exists():
        for csv_file in sorted(curves_dir.glob("*.csv")):
            fname = csv_file.stem
            # Determine curve type from filename
            sample_id, ct = parse_agent_curve_filename(fname)
            if not ct:
                continue

            # Read the CSV
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                continue

            # Extract x,y data
            x_col, y_col = get_column_names(ct)
            x_data = [float(r["x"]) for r in rows]
            y_data = [float(r["y"]) for r in rows]

            curve_entry = {
                "sample_id": sample_id,
                "curve_type": ct,
                "x": x_data,
                "y": y_data,
                "source_file": csv_file.name,
            }
            if ct not in curve_groups:
                curve_groups[ct] = []
            curve_groups[ct].append(curve_entry)

    # Write curves to subdirectories
    for ct, ct_curves in curve_groups.items():
        if ct not in VALID_TYPES:
            continue

        subdir = folder / ct
        subdir.mkdir(exist_ok=True)

        x_col, y_col = get_column_names(ct)

        # Group by sample_id - take the best curve per sample
        sample_curves: dict[str, dict] = {}
        for curve in ct_curves:
            sid = curve.get("sample_id", "unknown")
            x = curve.get("x", [])
            y = curve.get("y", [])
            if len(x) < 3:
                continue
            # Keep the curve with the most points for each sample
            if sid not in sample_curves or len(x) > len(sample_curves[sid].get("x", [])):
                sample_curves[sid] = curve

        for sid, curve in sample_curves.items():
            x = curve.get("x", [])
            y = curve.get("y", [])
            if not x or not y:
                continue

            out_file = subdir / f"{safe_filename(sid)}.csv"
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([x_col, y_col])
                for xi, yi in zip(x, y):
                    writer.writerow([xi, yi])

        print(f"  {ct}/: {len(sample_curves)} curve files")


def main():
    print("Converting agent_output to test format...")
    for folder_name in FOLDERS:
        print(f"\n{folder_name}:")
        process_folder(folder_name)
    print("\nDone!")


if __name__ == "__main__":
    main()
