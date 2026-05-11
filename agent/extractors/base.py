"""Base extractor class and extraction context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..llm_client import LLMClient
from ..config import AgentConfig
from ..output.schemas import Evidence


@dataclass
class ExtractionContext:
    """All preprocessed data for one paper folder."""
    folder: Path
    folder_name: str = ""
    sample_names: list[str] = field(default_factory=list)

    # Text
    main_text: str = ""
    si_text: str = ""

    # Tables: list of (page_num, table_as_list_of_lists)
    main_tables: list[tuple[int, list[list[str]]]] = field(default_factory=list)
    si_tables: list[tuple[int, list[list[str]]]] = field(default_factory=list)

    # Figure index: list of dicts from figure_index.csv
    figure_index: list[dict] = field(default_factory=list)

    # PDF paths
    main_pdf: Path | None = None
    si_pdf: Path | None = None

    # Evidence accumulator
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_evidence(self, field_name: str, value: str, source_type: str,
                     source_location: str, confidence: str = "medium", raw_text: str = ""):
        self.evidence.append(Evidence(
            field=field_name,
            value=str(value),
            source_type=source_type,
            source_location=source_location,
            confidence=confidence,
            raw_text=raw_text,
        ))

    def add_warning(self, msg: str):
        self.warnings.append(msg)


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self, llm: LLMClient, config: AgentConfig):
        self.llm = llm
        self.config = config

    @abstractmethod
    def extract(self, context: ExtractionContext) -> dict:
        """Extract data from context. Returns a dict with the extraction results."""
        pass
