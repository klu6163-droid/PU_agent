"""Prompts for curve digitization from plot images."""

SYSTEM_PROMPT = """You are an expert at reading scientific plots and extracting numerical data.
Digitize curves from plot images and return precise data points.
Always output valid JSON."""


def build_curve_prompt(
    plot_type: str,
    figure_caption: str,
    sample_labels: list[str],
    min_points: int = 100,
) -> str:
    """Build prompt for digitizing a specific plot type."""
    type_instructions = {
        "stress_strain": f"""This is a stress-strain curve for polyurethane elastomers.
- x-axis: strain (typically 0 to 500% or as fraction 0 to 5)
- y-axis: stress in MPa
- Extract all curves visible in the figure
- Include loading and unloading curves if present
- Extract at least {min_points} points for each primary curve when the image allows it""",
        "ftir": f"""This is an FTIR infrared spectrum.
- x-axis: wavenumber in cm^-1 (typically 4000 to 400, decreasing left to right)
- y-axis: absorbance or transmittance
- Extract the full spectrum with at least {min_points} points""",
        "saxs": f"""This is a SAXS (Small Angle X-ray Scattering) curve.
- x-axis: q in nm^-1 (typically 0.01 to 5)
- y-axis: intensity (often log scale)
- Extract the full curve with at least {min_points} points""",
        "waxs": f"""This is a WAXS (Wide Angle X-ray Scattering) curve.
- x-axis: 2 theta in degrees (typically 5 to 40) or q in nm^-1
- y-axis: intensity
- Extract the full curve with at least {min_points} points""",
        "dsc": f"""This is a DSC (Differential Scanning Calorimetry) curve.
- x-axis: temperature in degrees C (typically -100 to 250)
- y-axis: heat flow in mW/g or W/g
- Extract the full curve with at least {min_points} points""",
    }

    type_hint = type_instructions.get(plot_type, f"Extract all visible data from this {plot_type} plot.")
    labels_str = ", ".join(sample_labels) if sample_labels else "unknown"

    return f"""Digitize all curves from this scientific plot.

{type_hint}

## Known sample/curve labels: {labels_str}
## Figure caption: {figure_caption}

## Instructions:
1. Read the axis scales carefully from the image
2. Extract {min_points} data points per curve when possible; if there are many curves, 30-50 points each is acceptable
3. For each curve, provide the curve name/label as it appears in the legend
4. Ensure data points are in order (sorted by x)
5. Do not include grid lines or axis labels as data points
6. Keep the total response concise; prioritize accurate axes and representative points over excessive points

## Output JSON format:
{{
  "x_label": "strain",
  "y_label": "stress",
  "x_unit": "%",
  "y_unit": "MPa",
  "curves": [
    {{
      "name": "Sample-1",
      "data": [
        {{"x": 0.0, "y": 0.0}},
        {{"x": 10.0, "y": 1.5}}
      ]
    }}
  ]
}}
"""
