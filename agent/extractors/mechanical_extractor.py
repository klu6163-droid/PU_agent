"""Mechanical property extraction: text LLM extraction plus curve computation."""

from __future__ import annotations

import logging
import re

from .base import BaseExtractor, ExtractionContext
from ..output.schemas import MechanicalProperties, CurveData
from ..prompts.mechanical import SYSTEM_PROMPT, build_mechanical_prompt

logger = logging.getLogger(__name__)


def _normalize_unicode(text: str) -> str:
    """Replace common Unicode variants with ASCII equivalents."""
    replacements = {
        "\u2212": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u2248": "",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00b1": "+/-",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


def _safe_float(value) -> float | None:
    """Convert value to float, handling strings like 'N/A', '~50', and '50 +/- 5'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = _normalize_unicode(str(value).strip())
    if not s or s.lower() in ("n/a", "na", "-", "none", "null"):
        return None
    s = s.replace(",", "")
    s = re.split(r"\+/-|±", s, maxsplit=1)[0].strip()
    s = s.lstrip("~<>=")

    if ":" in s:
        parts = s.split(":")
        try:
            nums = [float(p) for p in parts]
            if len(nums) == 2 and nums[1] != 0:
                return nums[0] / nums[1]
        except ValueError:
            pass

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


class MechanicalExtractor(BaseExtractor):
    """Extract mechanical properties from text and curves."""

    def extract(self, context: ExtractionContext, curves: list[CurveData] | None = None) -> list[MechanicalProperties]:
        """Extract mechanical properties using text plus optional curve computation."""
        logger.info("  Extracting mechanical properties...")

        results = self._extract_via_llm(context)
        if curves:
            self._compute_from_curves(results, curves, context)

        logger.info(f"  Mechanical properties: {len(results)} samples")
        return results

    def _extract_via_llm(self, context: ExtractionContext) -> list[MechanicalProperties]:
        """Use LLM to extract mechanical properties from text."""
        full_text = context.main_text + "\n\n" + context.si_text

        table_text = self._tables_to_text(context)
        if table_text:
            full_text += "\n\n## Tables:\n" + table_text

        prompt = build_mechanical_prompt(full_text, context.sample_names)
        response = self.llm.send_text(prompt, system=SYSTEM_PROMPT)
        parsed = self.llm.parse_json_response(response)

        results = []
        if parsed and "mechanical" in parsed:
            for mdata in parsed["mechanical"]:
                mech = MechanicalProperties(
                    sample_id=mdata.get("sample_id", "unknown"),
                    young_modulus=_safe_float(mdata.get("young_modulus")),
                    tensile_strength=_safe_float(mdata.get("tensile_strength")),
                    elongation_at_break=_safe_float(mdata.get("elongation_at_break")),
                    toughness=_safe_float(mdata.get("toughness")),
                    hysteresis_loss=_safe_float(mdata.get("hysteresis_loss")),
                    resilience=_safe_float(mdata.get("resilience")),
                    fatigue_retention=_safe_float(mdata.get("fatigue_retention")),
                    strain_rate=_safe_float(mdata.get("strain_rate")),
                    test_temperature=_safe_float(mdata.get("test_temperature")),
                )
                results.append(mech)

                for field_name in [
                    "young_modulus",
                    "tensile_strength",
                    "elongation_at_break",
                    "toughness",
                    "hysteresis_loss",
                    "resilience",
                ]:
                    val = getattr(mech, field_name)
                    if val is not None:
                        context.add_evidence(
                            f"{mech.sample_id}.{field_name}",
                            str(val),
                            "llm_text",
                            "main_text",
                            confidence="high",
                        )

        existing_ids = {m.sample_id for m in results}
        for name in context.sample_names:
            if name not in existing_ids:
                results.append(MechanicalProperties(sample_id=name))

        return results

    def _compute_from_curves(
        self,
        results: list[MechanicalProperties],
        curves: list[CurveData],
        context: ExtractionContext,
    ):
        """Compute mechanical properties from stress-strain curves."""
        mech_map = {m.sample_id: m for m in results}

        sample_curves: dict[str, list[CurveData]] = {}
        for curve in curves:
            if curve.curve_type != "stress_strain" or len(curve.x) < 5:
                continue
            sample_id = curve.sample_id or curve.label
            matched_id = None
            if sample_id in mech_map:
                matched_id = sample_id
            else:
                for mid in mech_map:
                    if mid in sample_id or sample_id in mid:
                        matched_id = mid
                        break
            if matched_id:
                sample_curves.setdefault(matched_id, []).append(curve)

        for sample_id, matching_curves in sample_curves.items():
            mech = mech_map[sample_id]
            curve = max(matching_curves, key=lambda c: len(c.x))

            points = sorted(
                (float(x), float(y))
                for x, y in zip(curve.x, curve.y)
                if isinstance(x, (int, float)) and isinstance(y, (int, float))
            )
            if len(points) < 5:
                continue

            x = [p[0] for p in points]
            y = [p[1] for p in points]
            max_stress = max(y)
            max_strain = max(x)

            if mech.tensile_strength is None:
                mech.tensile_strength = round(max_stress, 1)
                context.add_evidence(
                    f"{mech.sample_id}.tensile_strength",
                    str(mech.tensile_strength),
                    "curve_computed",
                    curve.source_figure,
                    confidence="medium",
                )

            if mech.elongation_at_break is None:
                mech.elongation_at_break = round(max_strain, 1)
                context.add_evidence(
                    f"{mech.sample_id}.elongation_at_break",
                    str(mech.elongation_at_break),
                    "curve_computed",
                    curve.source_figure,
                    confidence="medium",
                )

            if mech.young_modulus is None and len(x) > 1:
                is_pct = max_strain > 10
                x_norm = [xi / 100.0 if is_pct else xi for xi in x]
                init_points = [(xi, yi) for xi, yi in zip(x_norm, y) if 0 < xi <= 0.05]

                confidence = "medium"
                if len(init_points) < 2:
                    nonzero = [(xi, yi) for xi, yi in zip(x_norm, y) if xi > 0]
                    init_points = nonzero[:min(5, len(nonzero))]
                    confidence = "low"

                if len(init_points) >= 2:
                    n = len(init_points)
                    sx = sum(p[0] for p in init_points)
                    sy = sum(p[1] for p in init_points)
                    sxx = sum(p[0] ** 2 for p in init_points)
                    sxy = sum(p[0] * p[1] for p in init_points)
                    denom = n * sxx - sx * sx
                    if abs(denom) > 1e-12:
                        slope = (n * sxy - sx * sy) / denom
                        if slope > 0:
                            mech.young_modulus = round(slope, 1)
                            context.add_evidence(
                                f"{mech.sample_id}.young_modulus",
                                str(mech.young_modulus),
                                "curve_computed",
                                curve.source_figure,
                                confidence=confidence,
                            )

            if mech.toughness is None and len(x) > 3:
                x_sorted = x
                y_sorted = y
                if max_strain > 10:
                    x_sorted = [xi / 100.0 for xi in x_sorted]

                toughness = 0.0
                for i in range(1, len(x_sorted)):
                    dx = x_sorted[i] - x_sorted[i - 1]
                    avg_y = (y_sorted[i] + y_sorted[i - 1]) / 2
                    toughness += dx * avg_y
                mech.toughness = round(toughness, 1)
                mech.toughness_source = "curve_computed"
                context.add_evidence(
                    f"{mech.sample_id}.toughness",
                    str(mech.toughness),
                    "curve_computed",
                    curve.source_figure,
                    confidence="medium",
                )

    def _tables_to_text(self, context: ExtractionContext) -> str:
        """Convert extracted tables to readable text for LLM."""
        parts = []
        for page_num, table in context.main_tables:
            parts.append(f"Table on page {page_num}:")
            for row in table[:20]:
                parts.append(" | ".join(str(c) for c in row if c))
        for page_num, table in context.si_tables:
            parts.append(f"SI Table on page {page_num}:")
            for row in table[:20]:
                parts.append(" | ".join(str(c) for c in row if c))
        return "\n".join(parts)
