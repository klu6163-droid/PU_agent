"""Prompts for literature and material composition extraction."""

SYSTEM_PROMPT = """You are an expert in polyurethane (PU) chemistry and materials science.
You extract structured data from scientific papers about polyurethane elastomers.
Always output valid JSON. Do not guess; use null for unknown values."""


def build_extraction_prompt(main_text: str, si_text: str, sample_names: list[str]) -> str:
    """Build prompt for extracting literature info and material composition."""
    samples_str = ", ".join(sample_names)

    return f"""Extract structured data from this polyurethane paper.

## Known sample names to extract: {samples_str}

## Instructions:
1. Extract article metadata (title, authors, journal, year, DOI)
2. For each sample listed above, extract material composition
3. If a field is not found in the text, use null (do not guess)
4. For SMILES strings, use standard chemical notation
5. hard_segment_content should be a percentage (e.g. 30.0 for 30%)
6. nco_oh_ratio is the molar ratio of NCO to OH groups

## Output JSON format:
{{
  "literature": {{
    "title": "...",
    "authors": ["Author1", "Author2"],
    "journal": "...",
    "year": 2024,
    "doi": "10.xxxx/xxxxx"
  }},
  "samples": [
    {{
      "sample_id": "SampleName",
      "soft_segment": "PCL",
      "soft_segment_smiles": "...",
      "soft_segment_mn": 2000.0,
      "diisocyanate": "MDI",
      "diisocyanate_smiles": "...",
      "chain_extender": "BDO",
      "chain_extender_smiles": "...",
      "hard_segment_content": 30.0,
      "soft_segment_ratio": null,
      "nco_oh_ratio": 1.5,
      "synthesis_method": "two-step",
      "processing_method": "solution casting",
      "notes": ""
    }}
  ]
}}

## Main text:
{main_text[:80000]}

## Supplementary Information:
{si_text[:40000]}
"""
