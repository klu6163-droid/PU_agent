"""Pydantic data models for all extraction outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LiteratureInfo(BaseModel):
    """Basic article metadata."""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    doi: str = ""
    article_id: str = ""
    pdf_filename: str = ""


class SampleInfo(BaseModel):
    """Material composition for a single sample."""
    sample_id: str
    soft_segment: str = "unknown"
    soft_segment_smiles: str = ""
    soft_segment_mn: float | None = None
    diisocyanate: str = "unknown"
    diisocyanate_smiles: str = ""
    chain_extender: str = "unknown"
    chain_extender_smiles: str = ""
    hard_segment_content: float | None = None
    soft_segment_ratio: float | None = None
    nco_oh_ratio: float | None = None
    synthesis_method: str | None = None
    processing_method: str | None = None
    notes: str = ""


class MechanicalProperties(BaseModel):
    """Mechanical properties for a single sample."""
    sample_id: str
    young_modulus: float | None = None
    tensile_strength: float | None = None
    elongation_at_break: float | None = None
    toughness: float | None = None
    hysteresis_loss: float | None = None
    resilience: float | None = None
    fatigue_retention: float | None = None
    strain_rate: float | None = None
    test_temperature: float | None = None
    toughness_source: str = ""


class CurveData(BaseModel):
    """A single extracted curve (e.g., one stress-strain curve for one sample)."""
    sample_id: str
    curve_type: str  # stress_strain, ftir, saxs, waxs, dsc
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    source_figure: str = ""
    source_page: int = 0
    source_pdf: str = ""
    confidence: str = "medium"
    label: str = ""  # curve label within the figure
    caption: str = ""
    extraction_method: str = "image_digitization"
    x_scale: str = "linear"
    y_scale: str = "linear"
    calibration: dict = Field(default_factory=dict)
    tick_marks: list[dict] = Field(default_factory=list)
    validation: dict = Field(default_factory=dict)
    review_notes: list[str] = Field(default_factory=list)
    artifact_paths: dict = Field(default_factory=dict)


class Evidence(BaseModel):
    """Provenance tracking for every extracted field."""
    field: str
    value: str
    source_type: str  # text_regex | table | llm_text | llm_image | curve_computed
    source_location: str  # e.g. "main_p3_Table1", "si_p15_FigureS18"
    confidence: str = "medium"  # high | medium | low
    raw_text: str = ""


class ExtractionResult(BaseModel):
    """Complete extraction result for one paper folder."""
    literature: LiteratureInfo = Field(default_factory=LiteratureInfo)
    samples: list[SampleInfo] = Field(default_factory=list)
    mechanical: list[MechanicalProperties] = Field(default_factory=list)
    curves: list[CurveData] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    folder_name: str = ""
