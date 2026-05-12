"""Agent orchestrator: coordinates all extractors for a folder."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import AgentConfig
from .llm_client import LLMClient
from .extractors.text_extractor import prepare_context
from .extractors.literature_extractor import LiteratureExtractor
from .extractors.curve_extractor import CurveExtractor
from .extractors.mechanical_extractor import MechanicalExtractor
from .output.schemas import ExtractionResult, LiteratureInfo
from .output.writer import write_results
from .output.report import generate_report

logger = logging.getLogger(__name__)

DEFAULT_STEPS = ("literature", "curves", "mechanical")


class PUExtractionAgent:
    """Main agent that orchestrates all extraction steps."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMClient(
            config.llm,
            retries=config.retries,
            backoff_base=config.backoff_base,
            timeout=config.api_timeout,
        )
        self.literature_extractor = LiteratureExtractor(self.llm, config)
        self.curve_extractor = CurveExtractor(self.llm, config)
        self.mechanical_extractor = MechanicalExtractor(self.llm, config)

    def process_folder(self, folder: Path, steps: list[str] | None = None) -> ExtractionResult:
        """Process a single paper folder end-to-end."""
        selected_steps = set(steps or DEFAULT_STEPS)
        logger.info(f"Processing {folder.name}...")

        # Step 1: Prepare context (text, tables, figure index)
        context = prepare_context(folder, self.config)

        if not context.main_text:
            logger.warning(f"  No main text found for {folder.name}, skipping")
            return ExtractionResult(folder_name=folder.name, warnings=["No main PDF found"])

        lit_result = {}
        if "literature" in selected_steps:
            logger.info("  [1/3] Literature + materials extraction...")
            lit_result = self.literature_extractor.extract(context)

        curves = []
        if "curves" in selected_steps:
            logger.info("  [2/3] Curve digitization...")
            curves = self.curve_extractor.extract(context)

        mechanical = []
        if "mechanical" in selected_steps:
            logger.info("  [3/3] Mechanical properties...")
            mechanical = self.mechanical_extractor.extract(context, curves)

        # Merge results
        result = ExtractionResult(
            literature=lit_result.get("literature") or LiteratureInfo(),
            samples=lit_result.get("samples", []),
            mechanical=mechanical,
            curves=curves,
            evidence=context.evidence,
            warnings=context.warnings,
            folder_name=folder.name,
        )

        logger.info(f"  Done: {len(result.samples)} samples, {len(result.curves)} curves, "
                     f"{len(result.mechanical)} mechanical entries")

        return result

    def run(self, folders: list[str] | None = None, steps: list[str] | None = None):
        """Run extraction on specified folders."""
        root = self.config.root_dir
        target_folders = folders or self.config.folders
        selected_steps = list(steps or DEFAULT_STEPS)

        logger.info("=" * 60)
        logger.info("PU Literature Data Extraction Agent")
        logger.info(f"Root: {root}")
        logger.info(f"Folders: {target_folders}")
        logger.info(f"Steps: {selected_steps}")
        logger.info("=" * 60)

        all_results = []

        for folder_name in target_folders:
            folder = root / folder_name
            if not folder.exists():
                logger.warning(f"Folder {folder} does not exist, skipping")
                continue

            result = self.process_folder(folder, selected_steps)
            all_results.append(result)

            # Write output
            output_dir = folder / self.config.output_dir_name
            write_results(output_dir, result)
            generate_report(output_dir, result)

        # Generate combined output
        if all_results:
            self._write_combined(all_results)

        logger.info("=" * 60)
        logger.info("Agent extraction complete!")
        logger.info("=" * 60)

    def _write_combined(self, results: list[ExtractionResult]):
        """Write combined output across all folders."""
        import csv

        combined_dir = self.config.root_dir / "combined_agent_output"
        combined_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("all_metadata.csv", "all_mechanical.csv", "all_warnings.md"):
            path = combined_dir / filename
            if path.exists():
                path.unlink()

        # Combined metadata
        all_samples = []
        for r in results:
            for s in r.samples:
                d = s.model_dump()
                d["source"] = r.folder_name
                all_samples.append(d)
        if all_samples:
            _write_csv(combined_dir / "all_metadata.csv", all_samples)

        # Combined mechanical
        all_mech = []
        for r in results:
            for m in r.mechanical:
                d = m.model_dump()
                d["source"] = r.folder_name
                all_mech.append(d)
        if all_mech:
            _write_csv(combined_dir / "all_mechanical.csv", all_mech)

        # Combined warnings
        with open(combined_dir / "all_warnings.md", "w", encoding="utf-8") as f:
            for r in results:
                if r.warnings:
                    f.write(f"## {r.folder_name}\n")
                    for w in r.warnings:
                        f.write(f"- {w}\n")
                    f.write("\n")

        logger.info(f"Combined output: {combined_dir}")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
