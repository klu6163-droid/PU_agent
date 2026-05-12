"""Fill missing mechanical and metadata fields from stress-strain curves and known data."""

import csv
import json
import os
import re
from pathlib import Path

PU_ROOT = Path("D:/CC Code/PU")
FOLDERS = [f"PU_{i:03d}" for i in range(1, 10)]

DEFAULT_MECHANICAL_FIELDS = [
    "sample_id",
    "young_modulus",
    "tensile_strength",
    "elongation_at_break",
    "toughness",
    "hysteresis_loss",
    "resilience",
    "fatigue_retention",
    "strain_rate",
    "test_temperature",
    "toughness_source",
]

# Known SMILES
SMILES = {
    "PCL": "O=C(OCCCCC)CCCCC(=O)O",
    "PTMEG": "OCCCOCCOCCOCCO",
    "PTMG": "OCCCOCCOCCOCCO",
    "PEG": "OCCOCCO",
    "PO3G": "OCC(C)CO",
    "MDI": "O=C=NC1=CC=C(C=C1)C2=CC=C(C=C2)N=C=O",
    "HDI": "O=C=NCCCCCCN=C=O",
    "IPDI": "CC1CC(C(=O)N=C=O)CC1C",
    "TDI": "CC1=CC(=CC=C1N=C=O)N=C=O",
    "HMDI": "O=C=NC1CCC(CC1)CC2CCC(CC2)N=C=O",
    "BDO": "OCCCCO",
    "EG": "OCCO",
    "EDA": "NCCN",
    "DMG": "OCCO",
    "SS": "NCCS",
    "ADH": "NNC(=O)CCCCC(=O)NN",
    "DD": "NNC(=O)CCCCCCCCCCC(=O)NN",
    "DABA": "NCCCC(N)C(=O)O",
    "carbohydrazide": "NNC(=O)NN",
}


def compute_mechanical(filepath: str) -> dict:
    """Compute mechanical properties from a stress-strain CSV."""
    try:
        with open(filepath, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except:
        return {}

    if len(rows) < 5:
        return {}

    # Find columns
    strain_col = stress_col = None
    for col in ['strain', 'x', 'X']:
        if col in rows[0]:
            strain_col = col
            break
    for col in ['stress', 'y', 'Y']:
        if col in rows[0]:
            stress_col = col
            break

    if not strain_col or not stress_col:
        return {}

    try:
        strain = [float(r[strain_col]) for r in rows]
        stress = [float(r[stress_col]) for r in rows]
    except:
        return {}

    if len(strain) < 5 or max(stress) <= 0:
        return {}

    # Normalize strain
    if max(strain) > 10:
        strain = [s / 100 for s in strain]

    props = {}

    # Tensile strength
    props['tensile_strength'] = round(max(stress), 1)

    # Elongation at break
    props['elongation_at_break'] = round(strain[-1] * 100, 1) if strain[-1] < 10 else round(strain[-1], 1)

    # Young's modulus (linear region)
    n_linear = max(5, len(strain) // 5)
    if n_linear >= 3:
        x_vals = strain[:n_linear]
        y_vals = stress[:n_linear]
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-10:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            if slope > 0:
                props['young_modulus'] = round(slope, 2)

    # Toughness
    toughness = 0
    for i in range(1, len(strain)):
        dx = strain[i] - strain[i - 1]
        dy = (stress[i] + stress[i - 1]) / 2
        toughness += dx * dy
    props['toughness'] = round(toughness, 1)

    return props


def parse_agent_curve_filename(filename: str) -> tuple[str, str]:
    """Parse sample_id and curve_type from agent_output/curves/ filename.

    Format: {sample_id}_{curve_type}_{source_figure}_{page}_{label}.csv
    Returns (sample_id, curve_type) or ("", "") if parsing fails.
    """
    name = filename.replace('.csv', '')
    curve_types = ['stress_strain', 'ftir', 'saxs', 'dsc', 'xrd', 'waxs', 'waxd']
    for ct in curve_types:
        idx = name.find('_' + ct + '_')
        if idx > 0:
            sample_id = name[:idx]
            return sample_id, ct
    return "", ""


def find_best_curve_from_agent(curves_dir: str, sample_id: str) -> str | None:
    """Find the best matching curve file from agent_output/curves/ for a sample ID."""
    if not os.path.isdir(curves_dir):
        return None

    files = [f for f in os.listdir(curves_dir) if f.endswith('.csv')]
    if not files:
        return None

    # Parse all filenames and find matches
    matches = []
    for f in files:
        sid, ct = parse_agent_curve_filename(f)
        if sid == sample_id and ct == 'stress_strain':
            matches.append(f)

    if not matches:
        # Try partial match
        sid_lower = sample_id.lower()
        for f in files:
            sid, ct = parse_agent_curve_filename(f)
            if ct == 'stress_strain' and (sid_lower in sid.lower() or sid.lower() in sid_lower):
                matches.append(f)

    if matches:
        # Pick the file with the most data points
        best = None
        best_points = 0
        for f in matches:
            filepath = os.path.join(curves_dir, f)
            try:
                with open(filepath, encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    n = sum(1 for _ in reader)
                    if n > best_points:
                        best_points = n
                        best = f
            except:
                pass
        if best:
            return os.path.join(curves_dir, best)

    return None


def find_curve_from_json(extraction_json: dict, sample_id: str) -> dict | None:
    """Find the best stress_strain curve for a sample_id from extraction_result.json."""
    curves = extraction_json.get('curves', [])
    matches = []
    for c in curves:
        if c.get('curve_type') == 'stress_strain' and c.get('sample_id') == sample_id:
            x = c.get('x', [])
            if len(x) >= 5:
                matches.append(c)

    if not matches:
        return None

    # Return the curve with the most points
    return max(matches, key=lambda c: len(c.get('x', [])))


def compute_mechanical_from_data(x_data: list, y_data: list) -> dict:
    """Compute mechanical properties from x (strain) and y (stress) data."""
    if len(x_data) < 5 or max(y_data) <= 0:
        return {}

    strain = [float(v) for v in x_data]
    stress = [float(v) for v in y_data]

    # Normalize strain
    if max(strain) > 10:
        strain = [s / 100 for s in strain]

    props = {}

    # Tensile strength
    props['tensile_strength'] = round(max(stress), 1)

    # Elongation at break
    props['elongation_at_break'] = round(strain[-1] * 100, 1) if strain[-1] < 10 else round(strain[-1], 1)

    # Young's modulus (linear region)
    n_linear = max(5, len(strain) // 5)
    if n_linear >= 3:
        x_vals = strain[:n_linear]
        y_vals = stress[:n_linear]
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-10:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            if slope > 0:
                props['young_modulus'] = round(slope, 2)

    # Toughness
    toughness = 0
    for i in range(1, len(strain)):
        dx = strain[i] - strain[i - 1]
        dy = (stress[i] + stress[i - 1]) / 2
        toughness += dx * dy
    props['toughness'] = round(toughness, 1)

    return props


def fill_smiles(sample: dict) -> dict:
    """Fill missing SMILES from known table."""
    for field, name_field in [
        ('soft_segment_smiles', 'soft_segment'),
        ('diisocyanate_smiles', 'diisocyanate'),
        ('chain_extender_smiles', 'chain_extender'),
    ]:
        if not sample.get(field, '').strip():
            name = sample.get(name_field, '').strip()
            if name in SMILES:
                sample[field] = SMILES[name]
    return sample


def process_folder(folder_name: str):
    """Fill missing data for one PU folder."""
    folder = PU_ROOT / folder_name

    # Read metadata
    meta_path = folder / "metadata.csv"
    if not meta_path.exists():
        return

    with open(meta_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        meta_rows = list(reader)
        meta_fields = reader.fieldnames

    # Read mechanical
    mech_path = folder / "mechanical.csv"
    mech_rows = []
    mech_fields = []
    if mech_path.exists():
        with open(mech_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            mech_rows = list(reader)
            mech_fields = reader.fieldnames
    if not mech_fields:
        mech_fields = DEFAULT_MECHANICAL_FIELDS.copy()

    # Load extraction_result.json for curve data
    extraction_json = {}
    result_path = folder / "agent_output" / "extraction_result.json"
    if result_path.exists():
        with open(result_path, encoding='utf-8') as f:
            extraction_json = json.load(f)

    curves_dir = str(folder / "agent_output" / "curves")
    updated_mech = 0
    updated_meta = 0

    # Build mechanical lookup by sample_id
    mech_by_id = {r['sample_id']: r for r in mech_rows if r.get('sample_id')}

    for sample in meta_rows:
        sid = sample.get('sample_id', '')
        if not sid:
            continue

        # Fill metadata SMILES
        orig_meta = dict(sample)
        sample = fill_smiles(sample)
        if sample != orig_meta:
            updated_meta += 1

        # Try to find curve data - first from extraction_result.json, then from CSV files
        computed = {}

        # Method 1: From extraction_result.json (has proper sample_ids)
        if extraction_json:
            curve = find_curve_from_json(extraction_json, sid)
            if curve:
                computed = compute_mechanical_from_data(curve['x'], curve['y'])

        # Method 2: From agent_output/curves/ CSV files
        if not computed:
            curve_file = find_best_curve_from_agent(curves_dir, sid)
            if curve_file:
                computed = compute_mechanical(curve_file)

        if not computed:
            continue

        # Update mechanical data - only fill EMPTY fields
        mech = mech_by_id.get(sid, {'sample_id': sid})
        changed = False
        for key, val in computed.items():
            if not mech.get(key, '').strip():
                mech[key] = str(val)
                changed = True

        if changed:
            mech_by_id[sid] = mech
            updated_mech += 1

    # Write updated metadata
    with open(meta_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields)
        writer.writeheader()
        writer.writerows(meta_rows)

    # Write updated mechanical
    mech_rows_out = [mech_by_id[sid] for sid in [r['sample_id'] for r in meta_rows] if sid in mech_by_id]
    # Add any mechanical rows not in metadata
    meta_ids = {r['sample_id'] for r in meta_rows}
    for sid, mech in mech_by_id.items():
        if sid not in meta_ids:
            mech_rows_out.append(mech)

    if mech_rows_out:
        with open(mech_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=mech_fields)
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in mech_fields} for row in mech_rows_out])

    print(f"  {folder_name}: updated {updated_meta} metadata, {updated_mech} mechanical")


def main():
    print("Filling missing data...")
    for folder_name in FOLDERS:
        print(f"\n{folder_name}:")
        process_folder(folder_name)
    print("\nDone!")


if __name__ == "__main__":
    main()
