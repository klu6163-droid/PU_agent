"""Image-based curve digitization and coordinate calibration.

The LLM is allowed to supply figure metadata, labels, and axis hints. Final x-y
points are produced here from vector paths or raster image analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..output.schemas import CurveData

logger = logging.getLogger(__name__)


@dataclass
class FigureMetadata:
    figure_id: str
    plot_type: str
    caption: str = ""
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    labels: list[str] = field(default_factory=list)
    legend: list[dict[str, Any]] = field(default_factory=list)
    axis_hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class DigitizationArtifacts:
    original_page: Path
    crop: Path
    mask: Path
    overlay: Path
    replotted: Path
    calibration: Path


@dataclass
class DigitizationResult:
    curves: list[CurveData]
    calibration: dict[str, Any]
    artifacts: DigitizationArtifacts
    warnings: list[str] = field(default_factory=list)


def fit_axis_model(ticks: list[dict[str, float]], axis: str) -> dict[str, Any]:
    """Fit linear and log10 mappings from pixel coordinate to axis value."""
    points = [
        (float(tick["pixel"]), float(tick["value"]))
        for tick in ticks
        if tick.get("pixel") is not None and tick.get("value") is not None
    ]
    if len(points) < 2:
        return {
            "axis": axis,
            "scale_type": "linear",
            "r2_linear": 0.0,
            "r2_log": 0.0,
            "coefficients": [1.0, 0.0],
            "reversed": False,
            "ticks": ticks,
        }

    pixels = np.array([p[0] for p in points], dtype=float)
    values = np.array([p[1] for p in points], dtype=float)
    linear_coef, linear_r2 = _fit_line(pixels, values)

    log_r2 = float("-inf")
    log_coef = [0.0, 0.0]
    if np.all(values > 0):
        log_coef, log_r2 = _fit_line(pixels, np.log10(values))

    scale_type = "log" if log_r2 > linear_r2 + 0.02 else "linear"
    coef = log_coef if scale_type == "log" else linear_coef
    return {
        "axis": axis,
        "scale_type": scale_type,
        "r2_linear": round(float(linear_r2), 6),
        "r2_log": round(float(log_r2 if math.isfinite(log_r2) else 0.0), 6),
        "coefficients": [round(float(coef[0]), 12), round(float(coef[1]), 12)],
        "reversed": bool(coef[0] < 0),
        "ticks": ticks,
    }


def map_pixel_to_value(pixel: float, calibration: dict[str, Any]) -> float:
    """Map one pixel coordinate to the calibrated physical value."""
    a, b = calibration["coefficients"]
    mapped = a * float(pixel) + b
    if calibration.get("scale_type") == "log":
        return float(10 ** mapped)
    return float(mapped)


def digitize_plot_image(
    page_image: np.ndarray,
    metadata: FigureMetadata,
    pdf_name: str,
    page_num: int,
    output_root: Path,
    min_points: int,
) -> DigitizationResult:
    """Digitize one figure from a rendered PDF page."""
    import cv2

    figure_slug = _safe_slug(metadata.figure_id)
    dirs = _ensure_artifact_dirs(output_root)
    original_path = dirs["figures"] / f"{figure_slug}_original.png"
    crop_path = dirs["figures"] / f"{figure_slug}_crop.png"
    mask_path = dirs["review"] / f"{figure_slug}_curve_mask.png"
    overlay_path = dirs["review"] / f"{figure_slug}_digitized_overlay.png"
    replotted_path = dirs["review"] / f"{figure_slug}_replotted_curve.png"
    calibration_path = dirs["calibration"] / f"{figure_slug}_axis_calibration.json"

    cv2.imwrite(str(original_path), page_image)
    crop_box = detect_plot_area(page_image)
    x0, y0, x1, y1 = crop_box
    crop = page_image[y0:y1, x0:x1].copy()
    cv2.imwrite(str(crop_path), crop)

    tick_candidates = ocr_ticks(crop)
    calibration = build_calibration(metadata, crop.shape, tick_candidates)
    calibration["pdf"] = pdf_name
    calibration["page"] = page_num
    calibration["figure_id"] = metadata.figure_id
    calibration["caption"] = metadata.caption
    calibration["crop_box"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    calibration["tick_recognition"] = tick_candidates

    vector_curves = extract_vector_paths_placeholder(metadata, calibration)
    method = "vector_path" if vector_curves else "opencv_raster_digitization"
    curves = vector_curves or extract_curves_from_raster(crop, metadata, calibration, min_points)

    mask = render_curve_mask(crop, curves, calibration)
    cv2.imwrite(str(mask_path), mask)
    overlay = render_overlay(crop, curves, calibration)
    cv2.imwrite(str(overlay_path), overlay)
    replotted = render_replotted(curves)
    cv2.imwrite(str(replotted_path), replotted)

    artifact_paths = {
        "original_page": str(original_path),
        "crop": str(crop_path),
        "curve_mask": str(mask_path),
        "overlay": str(overlay_path),
        "replotted": str(replotted_path),
        "calibration": str(calibration_path),
    }

    warnings: list[str] = []
    for curve in curves:
        curve.source_pdf = pdf_name
        curve.source_page = page_num
        curve.source_figure = metadata.figure_id
        curve.curve_type = metadata.plot_type
        curve.caption = metadata.caption
        curve.x_label = metadata.x_label
        curve.y_label = metadata.y_label
        curve.x_unit = metadata.x_unit
        curve.y_unit = metadata.y_unit
        curve.x_scale = calibration["x_axis"]["scale_type"]
        curve.y_scale = calibration["y_axis"]["scale_type"]
        curve.calibration = calibration
        curve.tick_marks = tick_candidates
        curve.extraction_method = method
        curve.artifact_paths = artifact_paths
        validation = validate_curve_physics(curve)
        curve.validation = validation
        curve.confidence = estimate_confidence(curve, calibration, validation, min_points)
        curve.review_notes = build_review_notes(curve, validation, min_points)
        warnings.extend(curve.review_notes)

    calibration["curves"] = [
        {
            "label": curve.label,
            "points": len(curve.x),
            "confidence": curve.confidence,
            "method": curve.extraction_method,
            "validation": curve.validation,
        }
        for curve in curves
    ]
    calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")

    return DigitizationResult(
        curves=curves,
        calibration=calibration,
        artifacts=DigitizationArtifacts(
            original_page=original_path,
            crop=crop_path,
            mask=mask_path,
            overlay=overlay_path,
            replotted=replotted_path,
            calibration=calibration_path,
        ),
        warnings=warnings,
    )


def detect_plot_area(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find a likely plot rectangle; fall back to a conservative page crop."""
    import cv2

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, int]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area > 0.08 * w * h and cw > 0.25 * w and ch > 0.2 * h:
            candidates.append((area, x, y, x + cw, y + ch))
    if candidates:
        _, x0, y0, x1, y1 = max(candidates, key=lambda item: item[0])
        pad = 20
        return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    return int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.88)


def ocr_ticks(crop: np.ndarray) -> list[dict[str, float]]:
    """Use optional OCR to read tick labels. Returns an empty list if unavailable."""
    try:
        import cv2
        import pytesseract
    except Exception:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception as exc:
        logger.debug("OCR tick detection unavailable: %s", exc)
        return []

    ticks: list[dict[str, float]] = []
    h, w = crop.shape[:2]
    for text, left, top, width, height, conf in zip(
        data.get("text", []),
        data.get("left", []),
        data.get("top", []),
        data.get("width", []),
        data.get("height", []),
        data.get("conf", []),
    ):
        value = _parse_number(text)
        if value is None:
            continue
        cx = float(left + width / 2)
        cy = float(top + height / 2)
        if cy > h * 0.75:
            axis = "x"
            pixel = cx
        elif cx < w * 0.25:
            axis = "y"
            pixel = cy
        else:
            continue
        ticks.append({"axis": axis, "pixel": pixel, "value": value, "confidence": _safe_float(conf, 0.0)})
    return ticks


def build_calibration(
    metadata: FigureMetadata,
    crop_shape: tuple[int, ...],
    tick_candidates: list[dict[str, float]],
) -> dict[str, Any]:
    """Build x/y calibration by fitting OCR ticks or domain fallback ticks."""
    h, w = crop_shape[:2]
    x_ticks = [tick for tick in tick_candidates if tick.get("axis") == "x"]
    y_ticks = [tick for tick in tick_candidates if tick.get("axis") == "y"]
    if len(x_ticks) < 2:
        x_ticks = default_ticks(metadata.plot_type, "x", w, h, metadata.axis_hints)
    if len(y_ticks) < 2:
        y_ticks = default_ticks(metadata.plot_type, "y", w, h, metadata.axis_hints)

    x_axis = fit_axis_model(x_ticks, "x")
    y_axis = fit_axis_model(y_ticks, "y")
    x_axis["label"] = metadata.x_label
    x_axis["unit"] = metadata.x_unit
    y_axis["label"] = metadata.y_label
    y_axis["unit"] = metadata.y_unit
    return {"x_axis": x_axis, "y_axis": y_axis}


def default_ticks(plot_type: str, axis: str, width: int, height: int, hints: dict[str, Any]) -> list[dict[str, float]]:
    """Domain-aware fallback ticks when OCR cannot recover tick labels."""
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    if plot_type == "stress_strain":
        x_min, x_max, y_min, y_max = 0.0, 500.0, 0.0, 50.0
    elif plot_type == "ftir":
        x_min, x_max, y_min, y_max = 4000.0, 400.0, 0.0, 1.0
    elif plot_type == "saxs":
        x_min, x_max, y_min, y_max = 0.01, 5.0, 1.0, 10000.0
    elif plot_type in ("xrd", "waxs"):
        x_min, x_max, y_min, y_max = 5.0, 60.0, 0.0, 1.0
    elif plot_type == "dsc":
        x_min, x_max, y_min, y_max = -100.0, 250.0, -1.0, 1.0

    axis_hint = hints.get(axis, {}) if isinstance(hints, dict) else {}
    v_min = _safe_float(axis_hint.get("min"), x_min if axis == "x" else y_min)
    v_max = _safe_float(axis_hint.get("max"), x_max if axis == "x" else y_max)
    if plot_type == "saxs" and axis == "x":
        return [
            {"axis": "x", "pixel": 0.0, "value": max(v_min, 1e-6), "confidence": 0.2, "source": "domain_default"},
            {"axis": "x", "pixel": float((width - 1) / 2), "value": math.sqrt(max(v_min, 1e-6) * max(v_max, 1e-6)), "confidence": 0.2, "source": "domain_default"},
            {"axis": "x", "pixel": float(width - 1), "value": max(v_max, 1e-6), "confidence": 0.2, "source": "domain_default"},
        ]
    if plot_type == "saxs" and axis == "y":
        return [
            {"axis": "y", "pixel": float(height - 1), "value": max(y_min, 1e-6), "confidence": 0.2, "source": "domain_default"},
            {"axis": "y", "pixel": float((height - 1) / 2), "value": math.sqrt(max(y_min, 1e-6) * max(y_max, 1e-6)), "confidence": 0.2, "source": "domain_default"},
            {"axis": "y", "pixel": 0.0, "value": max(y_max, 1e-6), "confidence": 0.2, "source": "domain_default"},
        ]

    if axis == "x":
        return [
            {"axis": "x", "pixel": 0.0, "value": v_min, "confidence": 0.2, "source": "domain_default"},
            {"axis": "x", "pixel": float(width - 1), "value": v_max, "confidence": 0.2, "source": "domain_default"},
        ]
    return [
        {"axis": "y", "pixel": float(height - 1), "value": v_min, "confidence": 0.2, "source": "domain_default"},
        {"axis": "y", "pixel": 0.0, "value": v_max, "confidence": 0.2, "source": "domain_default"},
    ]


def extract_vector_paths_placeholder(metadata: FigureMetadata, calibration: dict[str, Any]) -> list[CurveData]:
    """Reserved hook for PDF vector path extraction.

    PyMuPDF exposes page drawings, but robustly separating plotted data paths
    from axes/text requires the page coordinate crop. The pipeline keeps this
    hook first so vector extraction can be filled without changing callers.
    """
    return []


def extract_curves_from_raster(
    crop: np.ndarray,
    metadata: FigureMetadata,
    calibration: dict[str, Any],
    min_points: int,
) -> list[CurveData]:
    """Extract curves from a cropped plot using color clustering and components."""
    import cv2

    masks = color_cluster_masks(crop, max_clusters=max(2, min(6, len(metadata.labels) or 5)))
    components = []
    for mask in masks:
        mask = remove_axes_and_grid(mask, crop)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        components.extend(split_curve_components(mask, crop, max_curves=max(1, len(metadata.labels) or 5)))
    curves: list[CurveData] = []
    for idx, component_mask in enumerate(components, start=1):
        points = component_to_points(component_mask, calibration)
        if len(points) < 3:
            continue
        label = metadata.labels[idx - 1] if idx - 1 < len(metadata.labels) else f"curve_{idx}"
        if len(points) > min_points * 2:
            step = max(1, len(points) // max(min_points, 1))
            points = points[::step]
        curves.append(
            CurveData(
                sample_id=label,
                label=label,
                curve_type=metadata.plot_type,
                x=[round(p[0], 8) for p in points],
                y=[round(p[1], 8) for p in points],
            )
        )
    return curves


def color_cluster_masks(crop: np.ndarray, max_clusters: int = 5) -> list[np.ndarray]:
    """Build candidate curve masks with HSV/LAB clustering."""
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lightness = lab[:, :, 0]
    candidate = ((saturation > 35) & (value > 35)) | ((lightness < 80) & (saturation < 50))
    ys, xs = np.where(candidate)
    if len(xs) < 20:
        return [candidate.astype("uint8") * 255] if len(xs) else []

    features = np.column_stack([
        hsv[ys, xs, 0],
        hsv[ys, xs, 1],
        hsv[ys, xs, 2],
        lab[ys, xs, 1],
        lab[ys, xs, 2],
    ]).astype(np.float32)
    k = max(1, min(max_clusters, len(features) // 20))
    if k == 1:
        return [candidate.astype("uint8") * 255]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(features, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    masks: list[np.ndarray] = []
    for cluster_id in range(k):
        cluster_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cluster_points = labels.ravel() == cluster_id
        if int(np.count_nonzero(cluster_points)) < 20:
            continue
        cluster_mask[ys[cluster_points], xs[cluster_points]] = 255
        masks.append(cluster_mask)
    return masks or [candidate.astype("uint8") * 255]


def remove_axes_and_grid(mask: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Remove long horizontal/vertical axes and grid lines from a candidate mask."""
    import cv2

    cleaned = mask.copy()
    h, w = mask.shape[:2]
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, max(10, w // 8)), np.uint8))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(10, h // 8), 1), np.uint8))
    cleaned[horizontal > 0] = 0
    cleaned[vertical > 0] = 0
    return cleaned


def split_curve_components(mask: np.ndarray, crop: np.ndarray, max_curves: int = 5) -> list[np.ndarray]:
    """Combine HSV-like candidates with connected-component filtering."""
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[:2]
    components: list[tuple[int, np.ndarray]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < 20 or width < max(5, w * 0.03) or height < 2:
            continue
        component = np.zeros_like(mask)
        component[labels == label_id] = 255
        components.append((area, component))
    components.sort(key=lambda item: item[0], reverse=True)
    if not components and np.count_nonzero(mask):
        return [mask]
    return [component for _, component in components[:max_curves]]


def component_to_points(component_mask: np.ndarray, calibration: dict[str, Any]) -> list[tuple[float, float]]:
    """Skeleton-like reduction from mask pixels to one y value per x column."""
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0:
        return []
    points: list[tuple[float, float]] = []
    for px in sorted(set(int(x) for x in xs)):
        col_ys = ys[xs == px]
        if len(col_ys) == 0:
            continue
        py = float(np.median(col_ys))
        x_val = map_pixel_to_value(float(px), calibration["x_axis"])
        y_val = map_pixel_to_value(py, calibration["y_axis"])
        if math.isfinite(x_val) and math.isfinite(y_val):
            points.append((x_val, y_val))
    points.sort(key=lambda pair: pair[0])
    return points


def validate_curve_physics(curve: CurveData) -> dict[str, Any]:
    """Apply plot-type-specific physical sanity checks."""
    checks: dict[str, Any] = {"passed": True, "issues": []}
    x = curve.x
    y = curve.y
    if not x or not y:
        return {"passed": False, "issues": ["no digitized points"]}

    if curve.curve_type == "stress_strain":
        if any(v < -1e-9 for v in y):
            checks["issues"].append("stress contains negative values")
        if any(x[i] > x[i + 1] for i in range(len(x) - 1)):
            checks["issues"].append("strain is not monotonically increasing")
        checks["toughness_estimate"] = estimate_area(x, y)
    elif curve.curve_type == "ftir":
        checks["expected_x_range"] = "4000-400 cm^-1"
        checks["x_reversed"] = curve.calibration.get("x_axis", {}).get("reversed", False)
        if min(x) < 300 or max(x) > 4500:
            checks["issues"].append("wavenumber outside typical FTIR range")
    elif curve.curve_type == "saxs":
        if any(v <= 0 for v in x):
            checks["issues"].append("SAXS q must be positive")
        if any(v <= 0 for v in y):
            checks["issues"].append("SAXS intensity should be positive")
        if x:
            q_star = x[int(np.argmax(y))]
            checks["d_spacing"] = float(2 * math.pi / q_star) if q_star > 0 else None
    elif curve.curve_type in ("xrd", "waxs", "dsc"):
        if any(x[i] > x[i + 1] for i in range(len(x) - 1)):
            checks["issues"].append("x-axis should generally increase")

    checks["passed"] = not checks["issues"]
    return checks


def estimate_confidence(
    curve: CurveData,
    calibration: dict[str, Any],
    validation: dict[str, Any],
    min_points: int,
) -> str:
    """Estimate extraction confidence from point count, calibration, and checks."""
    score = 0
    if len(curve.x) >= min_points:
        score += 2
    elif len(curve.x) >= max(10, min_points // 2):
        score += 1
    if calibration["x_axis"].get("r2_linear", 0) > 0.98 or calibration["x_axis"].get("r2_log", 0) > 0.98:
        score += 1
    if calibration["y_axis"].get("r2_linear", 0) > 0.98 or calibration["y_axis"].get("r2_log", 0) > 0.98:
        score += 1
    if validation.get("passed"):
        score += 1
    return "high" if score >= 5 else "medium" if score >= 3 else "low"


def build_review_notes(curve: CurveData, validation: dict[str, Any], min_points: int) -> list[str]:
    """Generate human review recommendations."""
    notes: list[str] = []
    if len(curve.x) < min_points:
        notes.append(f"{curve.label} has {len(curve.x)} points; below target {min_points}")
    for issue in validation.get("issues", []):
        notes.append(f"{curve.label}: {issue}")
    if curve.confidence == "low":
        notes.append(f"{curve.label}: low confidence; inspect overlay and calibration JSON")
    return notes


def render_curve_mask(crop: np.ndarray, curves: list[CurveData], calibration: dict[str, Any]) -> np.ndarray:
    """Render extracted curve pixels as a binary mask for review."""
    import cv2

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for curve in curves:
        pixels = values_to_pixels(curve.x, curve.y, calibration)
        for px, py in pixels:
            cv2.circle(mask, (px, py), 1, 255, -1)
    return mask


def render_overlay(crop: np.ndarray, curves: list[CurveData], calibration: dict[str, Any]) -> np.ndarray:
    """Overlay extracted points on the cropped original image."""
    import cv2

    overlay = crop.copy()
    colors = [(0, 0, 255), (0, 180, 0), (255, 0, 0), (0, 180, 180), (180, 0, 180)]
    for idx, curve in enumerate(curves):
        pixels = values_to_pixels(curve.x, curve.y, calibration)
        color = colors[idx % len(colors)]
        for p1, p2 in zip(pixels, pixels[1:]):
            cv2.line(overlay, p1, p2, color, 2)
    return overlay


def render_replotted(curves: list[CurveData], width: int = 900, height: int = 650) -> np.ndarray:
    """Render a simple replot of extracted x-y data without matplotlib."""
    import cv2

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if not curves:
        cv2.putText(canvas, "No curves extracted", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return canvas
    xs = [v for curve in curves for v in curve.x]
    ys = [v for curve in curves for v in curve.y]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    left, right, top, bottom = 80, width - 40, 40, height - 80
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)
    colors = [(0, 0, 255), (0, 180, 0), (255, 0, 0), (0, 180, 180), (180, 0, 180)]
    for idx, curve in enumerate(curves):
        pts = []
        for x, y in zip(curve.x, curve.y):
            px = int(left + (x - x_min) / (x_max - x_min) * (right - left))
            py = int(bottom - (y - y_min) / (y_max - y_min) * (bottom - top))
            pts.append((px, py))
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(canvas, p1, p2, colors[idx % len(colors)], 2)
    return canvas


def values_to_pixels(x_values: list[float], y_values: list[float], calibration: dict[str, Any]) -> list[tuple[int, int]]:
    """Invert calibration approximately for review rendering."""
    pixels: list[tuple[int, int]] = []
    for x, y in zip(x_values, y_values):
        px = _value_to_pixel(x, calibration["x_axis"])
        py = _value_to_pixel(y, calibration["y_axis"])
        pixels.append((int(round(px)), int(round(py))))
    return pixels


def _value_to_pixel(value: float, axis_calibration: dict[str, Any]) -> float:
    a, b = axis_calibration["coefficients"]
    target = math.log10(value) if axis_calibration.get("scale_type") == "log" and value > 0 else value
    if abs(a) < 1e-12:
        return 0.0
    return (target - b) / a


def estimate_area(x: list[float], y: list[float]) -> float:
    """Trapezoidal area under a curve."""
    area = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        area += dx * (y[i] + y[i - 1]) / 2
    return round(float(area), 6)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[list[float], float]:
    coef = np.polyfit(x, y, 1)
    predicted = np.polyval(coef, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return [float(coef[0]), float(coef[1])], float(r2)


def _parse_number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_") or "figure"


def _ensure_artifact_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "figures": output_root / "figures",
        "curves": output_root / "curves",
        "review": output_root / "review",
        "calibration": output_root / "calibration",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
