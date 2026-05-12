"""Curve extraction using LLM-assisted localization and image digitization."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import numpy as np

from .base import BaseExtractor, ExtractionContext
from .curve_digitizer import FigureMetadata, digitize_plot_image
from ..output.schemas import CurveData
from ..prompts.curve_digitize import SYSTEM_PROMPT, build_curve_metadata_prompt

logger = logging.getLogger(__name__)

TARGET_TYPES = {"stress_strain", "ftir", "saxs", "waxs", "xrd", "dsc"}


def _image_to_base64(image: np.ndarray, max_dim: int = 1600) -> str:
    """Convert numpy image to base64 PNG for LLM metadata inspection."""
    import cv2

    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf).decode("utf-8")


class CurveExtractor(BaseExtractor):
    """Extract curve data without using LLM-generated x-y values."""

    def extract(self, context: ExtractionContext) -> list[CurveData]:
        """Process target figures with metadata LLM + vector/raster digitization."""
        all_curves: list[CurveData] = []
        figures_to_process = self._select_target_figures(context)
        logger.info(f"  Processing {len(figures_to_process)} figures for curve extraction...")

        output_root = context.folder / self.config.output_dir_name
        for fig in figures_to_process:
            pdf_path = context.main_pdf if fig["pdf_type"] == "main" else context.si_pdf
            if not pdf_path:
                continue
            page_num = int(fig.get("page_number", 0))
            image = self._render_page(pdf_path, page_num)
            if image is None:
                context.add_warning(f"Could not render {pdf_path.name} page {page_num}")
                continue

            plot_type = fig["_target_type"]
            metadata = self._read_figure_metadata(image, fig, plot_type)
            logger.info(f"    {metadata.figure_id} ({metadata.plot_type}) from {pdf_path.name} p{page_num}")

            result = digitize_plot_image(
                page_image=image,
                metadata=metadata,
                pdf_name=pdf_path.name,
                page_num=page_num,
                output_root=output_root,
                min_points=self.config.min_curve_points,
            )

            for warning in result.warnings:
                context.add_warning(warning)

            for curve in result.curves:
                if len(curve.x) < 3:
                    context.add_warning(f"Curve {curve.label} has only {len(curve.x)} points, skipped")
                    continue
                all_curves.append(curve)
                context.add_evidence(
                    f"{curve.label}.curve",
                    f"{len(curve.x)} points",
                    curve.extraction_method,
                    f"{curve.source_pdf}:p{curve.source_page}:{curve.source_figure}",
                    confidence=curve.confidence,
                    raw_text=curve.caption,
                )
                logger.info(
                    "      Extracted: %s (%s points, %s, %s/%s)",
                    curve.label,
                    len(curve.x),
                    curve.confidence,
                    curve.x_scale,
                    curve.y_scale,
                )

        logger.info(f"  Total curves extracted: {len(all_curves)}")
        return all_curves

    def _select_target_figures(self, context: ExtractionContext) -> list[dict]:
        """Use text-derived figure index to select candidate figures."""
        selected: list[dict] = []
        for fig in context.figure_index:
            plot_types = fig.get("plot_types", [])
            for ptype in plot_types:
                normalized = "xrd" if ptype == "waxs" else ptype
                if normalized in TARGET_TYPES:
                    selected.append({**fig, "_target_type": normalized})

        return selected

    def _read_figure_metadata(self, image: np.ndarray, fig: dict, plot_type: str) -> FigureMetadata:
        """Ask the LLM for metadata only; never accept x-y data from the LLM."""
        prompt = build_curve_metadata_prompt(
            plot_type=plot_type,
            figure_caption=fig.get("caption", ""),
            sample_labels=fig.get("sample_labels", []),
        )
        parsed = None
        try:
            response = self.llm.send_image(_image_to_base64(image, self.config.max_image_dim), prompt, system=SYSTEM_PROMPT)
            parsed = self.llm.parse_json_response(response)
        except Exception as exc:
            logger.warning("LLM metadata read failed for %s: %s", fig.get("figure_id", "figure"), exc)

        if not isinstance(parsed, dict):
            parsed = {}

        if "curves" in parsed or "data" in parsed:
            logger.warning("Ignoring prohibited LLM curve data for %s", fig.get("figure_id", "figure"))

        labels = parsed.get("curve_labels") or parsed.get("sample_labels") or fig.get("sample_labels", [])
        if not isinstance(labels, list):
            labels = []

        return FigureMetadata(
            figure_id=str(parsed.get("figure_id") or fig.get("figure_id", "Figure")),
            plot_type=_normalize_plot_type(str(parsed.get("plot_type") or plot_type)),
            caption=str(parsed.get("caption") or fig.get("caption", "")),
            x_label=str(parsed.get("x_axis", {}).get("label", "")) if isinstance(parsed.get("x_axis"), dict) else "",
            y_label=str(parsed.get("y_axis", {}).get("label", "")) if isinstance(parsed.get("y_axis"), dict) else "",
            x_unit=str(parsed.get("x_axis", {}).get("unit", "")) if isinstance(parsed.get("x_axis"), dict) else "",
            y_unit=str(parsed.get("y_axis", {}).get("unit", "")) if isinstance(parsed.get("y_axis"), dict) else "",
            labels=[str(label) for label in labels],
            legend=parsed.get("legend", []) if isinstance(parsed.get("legend"), list) else [],
            axis_hints={
                "x": parsed.get("x_axis", {}) if isinstance(parsed.get("x_axis"), dict) else {},
                "y": parsed.get("y_axis", {}) if isinstance(parsed.get("y_axis"), dict) else {},
            },
        )

    def _render_page(self, pdf_path: Path, page_num: int) -> np.ndarray | None:
        """Render a PDF page to a BGR numpy image."""
        try:
            import cv2
            import fitz

            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None
            page = doc[page_num - 1]
            zoom = self.config.render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            doc.close()

            if pix.n == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            if pix.n == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            logger.warning("Failed to render %s p%s: %s", pdf_path.name, page_num, exc)
            return None


def _normalize_plot_type(plot_type: str) -> str:
    normalized = plot_type.strip().lower().replace("-", "_")
    aliases = {"waxs": "xrd", "waxd": "xrd", "x_ray_diffraction": "xrd"}
    return aliases.get(normalized, normalized)
