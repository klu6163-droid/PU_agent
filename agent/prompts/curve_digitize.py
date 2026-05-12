"""Prompts for LLM-assisted figure metadata reading.

The LLM must not provide final x-y curve data. Numerical curves are extracted
by vector/image algorithms in agent.extractors.curve_digitizer.
"""

SYSTEM_PROMPT = """You are an expert at identifying scientific plot metadata.
You may locate figures, read captions, axes, units, legends, and curve labels.
Do not digitize curves and do not output x-y data points. Always output valid JSON."""


def build_curve_metadata_prompt(
    plot_type: str,
    figure_caption: str,
    sample_labels: list[str],
) -> str:
    """Build a prompt that asks for metadata only, not curve points."""
    labels_str = ", ".join(sample_labels) if sample_labels else "unknown"
    return f"""Inspect this scientific plot page and return metadata only.

## Target plot type
{plot_type}

## Text-derived caption
{figure_caption}

## Known sample names
{labels_str}

## Allowed tasks
1. Decide whether the page contains the target figure.
2. Identify plot type, figure number/panel, and caption.
3. Read x/y axis names and units.
4. Associate legend entries, sample names, curve labels, marker styles, and line styles.
5. Note visible tick labels or axis min/max hints if readable.
6. Summarize relevant nearby text printed in the figure.

## Forbidden tasks
- Do not output curve x-y data.
- Do not estimate final numerical data points.
- Do not create any field containing digitized point arrays.

## Output JSON schema
{{
  "contains_target": true,
  "figure_id": "Figure 3a",
  "plot_type": "{plot_type}",
  "caption": "caption text",
  "x_axis": {{
    "label": "strain",
    "unit": "%",
    "min": 0,
    "max": 500,
    "tick_labels": ["0", "100", "200"],
    "direction_hint": "increasing"
  }},
  "y_axis": {{
    "label": "stress",
    "unit": "MPa",
    "min": 0,
    "max": 50,
    "tick_labels": ["0", "10", "20"],
    "direction_hint": "increasing"
  }},
  "curve_labels": ["Sample A", "Sample B"],
  "legend": [
    {{"label": "Sample A", "color": "red", "marker": "circle", "line_style": "solid"}}
  ],
  "text_context": "short description"
}}
"""


def build_curve_prompt(*args, **kwargs) -> str:
    """Compatibility wrapper; returns metadata-only prompt."""
    kwargs.pop("min_points", None)
    return build_curve_metadata_prompt(*args, **kwargs)
