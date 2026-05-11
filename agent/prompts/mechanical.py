"""Prompts for mechanical property extraction from text."""

SYSTEM_PROMPT = """You are an expert in polymer mechanical testing.
Extract mechanical properties from scientific paper text and tables.
Always output valid JSON. Use null for unknown values; never guess."""


def build_mechanical_prompt(text: str, sample_names: list[str]) -> str:
    """Build prompt for extracting mechanical properties from text."""
    samples_str = ", ".join(sample_names)

    return f"""Extract mechanical properties for each sample from this text and tables.

## Samples: {samples_str}

## Properties to extract:
- young_modulus: Young's modulus in MPa (often called "initial modulus" or "E")
- tensile_strength: in MPa (often called "stress at break" or "sigma_b")
- elongation_at_break: in % (often called "strain at break" or "epsilon_b")
- toughness: in MJ/m^3 (area under stress-strain curve, often explicitly stated)
- hysteresis_loss: in %
- resilience: in %
- fatigue_retention: in %
- strain_rate: in mm/min or min^-1
- test_temperature: in degrees C

## Important:
1. Look carefully at tables; mechanical data is often in Table 1 or Table S1
2. Check both main text and supporting information
3. Values may be given as ranges (use the average) or mean +/- std (use the mean)
4. If toughness is not explicitly stated but stress-strain data is given, leave it null (we compute it from curves)
5. Report all samples, even if some fields are null

## Output JSON format:
{{
  "mechanical": [
    {{
      "sample_id": "SampleName",
      "young_modulus": 50.0,
      "tensile_strength": 30.0,
      "elongation_at_break": 500.0,
      "toughness": 100.0,
      "hysteresis_loss": null,
      "resilience": null,
      "fatigue_retention": null,
      "strain_rate": 100,
      "test_temperature": 25
    }}
  ]
}}

## Text:
{text[:60000]}
"""
