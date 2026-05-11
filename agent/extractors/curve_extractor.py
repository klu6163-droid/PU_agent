"""Curve digitization via multimodal LLM."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

import numpy as np

from .base import BaseExtractor, ExtractionContext
from ..output.schemas import CurveData
from ..prompts.curve_digitize import SYSTEM_PROMPT, build_curve_prompt

logger = logging.getLogger(__name__)

TARGET_TYPES = {"stress_strain", "ftir", "saxs", "waxs", "dsc"}


def _image_to_base64(image: np.ndarray, max_dim: int = 1600) -> str:
    """Convert numpy image to base64 PNG, resizing if needed."""
    import cv2

    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf).decode("utf-8")


def _as_float(value) -> float | None:
    """Convert a parsed point coordinate to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CurveExtractor(BaseExtractor):
    """Extract curve data from figures using multimodal LLM."""

    def extract(self, context: ExtractionContext) -> list[CurveData]:
        """Process all figures in the context, digitize curves via API."""
        all_curves: list[CurveData] = []

        # Group figures by (pdf_type, page_number) to avoid rendering same page twice
        pages_to_process: dict[tuple[str, int], list[dict]] = {}
        for fig in context.figure_index:
            for ptype in fig.get("plot_types", []):
                if ptype in TARGET_TYPES:
                    key = (fig["pdf_type"], fig["page_number"])
                    if key not in pages_to_process:
                        pages_to_process[key] = []
                    pages_to_process[key].append({**fig, "_target_type": ptype})

        logger.info(f"  Processing {len(pages_to_process)} pages for curve extraction...")

        for (pdf_type, page_num), figures in pages_to_process.items():
            pdf_path = context.main_pdf if pdf_type == "main" else context.si_pdf
            if not pdf_path:
                continue

            # Render page image
            image = self._render_page(pdf_path, page_num)
            if image is None:
                continue

            image_b64 = _image_to_base64(image, self.config.max_image_dim)

            for fig in figures:
                plot_type = fig["_target_type"]
                caption = fig.get("caption", "")
                sample_labels = fig.get("sample_labels", [])

                logger.info(f"    {fig['figure_id']} ({plot_type}) from {pdf_path.name} p{page_num}")

                curves = self._digitize_figure(
                    image_b64, plot_type, caption, sample_labels,
                    pdf_path.name, page_num, fig["figure_id"],
                )

                for curve in curves:
                    point_count = len(curve.x)
                    if point_count < 3:
                        context.add_warning(f"Curve {curve.label} has only {point_count} points, skipped")
                        continue
                    if point_count < self.config.min_curve_points:
                        context.add_warning(
                            f"Curve {curve.label} has {point_count} points; "
                            f"below target {self.config.min_curve_points}"
                        )
                    all_curves.append(curve)
                    logger.info(f"      Extracted: {curve.label} ({point_count} points)")

        logger.info(f"  Total curves extracted: {len(all_curves)}")
        return all_curves

    def _render_page(self, pdf_path: Path, page_num: int) -> np.ndarray | None:
        """Render a PDF page to numpy array."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None
            page = doc[page_num - 1]  # 0-indexed
            zoom = self.config.render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            doc.close()

            import cv2
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            logger.warning(f"Failed to render {pdf_path.name} p{page_num}: {e}")
            return None

    def _digitize_figure(
        self,
        image_b64: str,
        plot_type: str,
        caption: str,
        sample_labels: list[str],
        pdf_name: str,
        page_num: int,
        figure_id: str,
    ) -> list[CurveData]:
        """Digitize curves from one figure via multimodal API."""
        prompt = build_curve_prompt(
            plot_type=plot_type,
            figure_caption=caption,
            sample_labels=sample_labels,
            min_points=self.config.min_curve_points,
        )

        response = self.llm.send_image(image_b64, prompt, system=SYSTEM_PROMPT)
        parsed = self.llm.parse_json_response(response)

        if not parsed or "curves" not in parsed:
            return []

        curves = []
        for curve_data in parsed["curves"]:
            data_points = curve_data.get("data", [])
            x_vals = []
            y_vals = []
            for point in data_points:
                x = _as_float(point.get("x"))
                y = _as_float(point.get("y"))
                if x is None or y is None:
                    continue
                x_vals.append(x)
                y_vals.append(y)

            if not x_vals:
                continue

            point_count = len(x_vals)
            if point_count >= self.config.min_curve_points:
                confidence = "high"
            elif point_count >= max(10, self.config.min_curve_points // 2):
                confidence = "medium"
            else:
                confidence = "low"

            curves.append(CurveData(
                sample_id=curve_data.get("name", "unknown"),
                curve_type=plot_type,
                x=x_vals,
                y=y_vals,
                x_label=parsed.get("x_label", ""),
                y_label=parsed.get("y_label", ""),
                x_unit=parsed.get("x_unit", ""),
                y_unit=parsed.get("y_unit", ""),
                source_figure=figure_id,
                source_page=page_num,
                source_pdf=pdf_name,
                confidence=confidence,
                label=curve_data.get("name", ""),
            ))

        return curves
