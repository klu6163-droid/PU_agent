"""Literature + material composition extraction via LLM full-text understanding."""

from __future__ import annotations

import logging
import re

from .base import BaseExtractor, ExtractionContext
from ..output.schemas import LiteratureInfo, SampleInfo
from ..prompts.literature import SYSTEM_PROMPT, build_extraction_prompt

logger = logging.getLogger(__name__)


def _safe_float(value) -> float | None:
    """Convert value to float, handling strings like '1:1', 'N/A', etc."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("n/a", "na", "-", "none", "null"):
        return None
    # Handle ratio format like "1:1" -> 1.0, "2:1" -> 2.0
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [float(p) for p in parts]
            if len(nums) == 2 and nums[1] != 0:
                return nums[0] / nums[1]
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return None


# Known SMILES for common PU components
KNOWN_SMILES = {
    "PCL": "O=C(OCCCCC)CCCCC(=O)O",
    "PTMEG": "OCCCOCCOCCOCCO",
    "PTMG": "OCCCOCCOCCOCCO",
    "PEG": "OCCOCCO",
    "PO3G": "OCC(C)CO",
    "MDI": "O=C=NC1=CC=C(C=C1)C2=CC=C(C=C2)N=C=O",
    "HDI": "O=C=NCCCCCCN=C=O",
    "IPDI": "CC1CC(C(=O)N=C=O)CC1C",
    "TDI": "CC1=CC(=CC=C1N=C=O)N=C=O",
    "BDO": "OCCCCO",
    "EG": "OCCO",
    "EDA": "NCCN",
    "1,4-BDO": "OCCCCO",
    "DMG": "OCCO",
    "SS": "NCCS",
    "DTDA": "",
}


class LiteratureExtractor(BaseExtractor):
    """Extract literature info and material composition using LLM."""

    def extract(self, context: ExtractionContext) -> dict:
        """Send full text to LLM, get structured extraction."""
        logger.info("  Extracting literature + materials via LLM...")

        prompt = build_extraction_prompt(
            context.main_text,
            context.si_text,
            context.sample_names,
        )

        response = self.llm.send_text(prompt, system=SYSTEM_PROMPT)
        parsed = self.llm.parse_json_response(response)

        if not parsed:
            logger.warning("  LLM extraction failed, falling back to regex")
            return self._fallback_extract(context)

        # Parse literature info
        lit_data = parsed.get("literature", {})
        literature = LiteratureInfo(
            title=lit_data.get("title", ""),
            authors=lit_data.get("authors", []),
            journal=lit_data.get("journal", ""),
            year=lit_data.get("year"),
            doi=lit_data.get("doi", ""),
            article_id=context.folder_name,
            pdf_filename=context.main_pdf.name if context.main_pdf else "",
        )

        # Parse sample info
        samples = []
        for sdata in parsed.get("samples", []):
            sample = SampleInfo(
                sample_id=str(sdata.get("sample_id", "unknown")),
                soft_segment=str(sdata.get("soft_segment") or "unknown"),
                soft_segment_smiles=str(sdata.get("soft_segment_smiles") or ""),
                soft_segment_mn=_safe_float(sdata.get("soft_segment_mn")),
                diisocyanate=str(sdata.get("diisocyanate") or "unknown"),
                diisocyanate_smiles=str(sdata.get("diisocyanate_smiles") or ""),
                chain_extender=str(sdata.get("chain_extender") or "unknown"),
                chain_extender_smiles=str(sdata.get("chain_extender_smiles") or ""),
                hard_segment_content=_safe_float(sdata.get("hard_segment_content")),
                soft_segment_ratio=_safe_float(sdata.get("soft_segment_ratio")),
                nco_oh_ratio=_safe_float(sdata.get("nco_oh_ratio")),
                synthesis_method=sdata.get("synthesis_method"),
                processing_method=sdata.get("processing_method"),
                notes=str(sdata.get("notes") or ""),
            )

            # Fill missing SMILES from known table
            sample = self._fill_missing_smiles(sample)

            # Add evidence
            context.add_evidence(
                "composition", sample.sample_id,
                "llm_text", "main_text_full",
                confidence="high",
            )

            samples.append(sample)

        # Ensure all molecule.txt names have a sample entry
        existing_ids = {s.sample_id for s in samples}
        for name in context.sample_names:
            if name not in existing_ids:
                samples.append(SampleInfo(sample_id=name))
                context.add_warning(f"Sample {name} not found in LLM extraction")

        logger.info(f"  Extracted: {literature.title[:60]}..., {len(samples)} samples")
        return {"literature": literature, "samples": samples}

    def _fill_missing_smiles(self, sample: SampleInfo) -> SampleInfo:
        """Fill missing SMILES from known table."""
        if not sample.soft_segment_smiles and sample.soft_segment in KNOWN_SMILES:
            sample.soft_segment_smiles = KNOWN_SMILES[sample.soft_segment]
        if not sample.diisocyanate_smiles and sample.diisocyanate in KNOWN_SMILES:
            sample.diisocyanate_smiles = KNOWN_SMILES[sample.diisocyanate]
        if not sample.chain_extender_smiles and sample.chain_extender in KNOWN_SMILES:
            sample.chain_extender_smiles = KNOWN_SMILES[sample.chain_extender]
        return sample

    def _fallback_extract(self, context: ExtractionContext) -> dict:
        """Regex-based fallback extraction."""
        from ..output.schemas import LiteratureInfo, SampleInfo

        literature = LiteratureInfo(
            article_id=context.folder_name,
            pdf_filename=context.main_pdf.name if context.main_pdf else "",
        )

        # Try to extract title from first page
        if context.main_text:
            lines = context.main_text.split("\n")[:30]
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.startswith("---"):
                    literature.title = line
                    break

        samples = []
        for name in context.sample_names:
            sample = SampleInfo(sample_id=name)
            # Simple keyword matching
            full_text = context.main_text + " " + context.si_text
            for kw, info in [
                ("PCL", ("PCL", "PCL")),
                ("PTMEG", ("PTMEG", "PTMEG")),
                ("PTMG", ("PTMEG", "PTMG")),
                ("PEG", ("PEG", "PEG")),
                ("MDI", ("MDI", "MDI")),
                ("HDI", ("HDI", "HDI")),
                ("IPDI", ("IPDI", "IPDI")),
                ("BDO", ("BDO", "BDO")),
                ("EG", ("EG", "EG")),
            ]:
                if kw in full_text:
                    if info[0] in ("PCL", "PTMEG", "PTMG", "PEG", "PO3G"):
                        sample.soft_segment = info[1]
                    elif info[0] in ("MDI", "HDI", "IPDI", "TDI"):
                        sample.diisocyanate = info[1]
                    elif info[0] in ("BDO", "EG", "EDA"):
                        sample.chain_extender = info[1]

            sample = self._fill_missing_smiles(sample)
            samples.append(sample)

        return {"literature": literature, "samples": samples}
