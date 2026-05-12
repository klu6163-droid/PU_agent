from pathlib import Path

from agent.config import load_config
from agent.extractors.curve_digitizer import fit_axis_model, map_pixel_to_value
from agent.llm_client import LLMClient
from agent.output.schemas import CurveData, ExtractionResult
from agent.output.writer import write_results
from agent.prompts.curve_digitize import build_curve_metadata_prompt
from agent.utils_pdf import classify_pdf


def test_parse_json_response_from_markdown_block():
    text = """```json
{"mechanical": [{"sample_id": "A", "young_modulus": 1.2}]}
```"""

    parsed = LLMClient.parse_json_response(text)

    assert parsed["mechanical"][0]["sample_id"] == "A"
    assert parsed["mechanical"][0]["young_modulus"] == 1.2


def test_write_results_removes_stale_curve_files(tmp_path: Path):
    output_dir = tmp_path / "agent_output"
    stale_curve_dir = output_dir / "curves"
    stale_curve_dir.mkdir(parents=True)
    stale_file = stale_curve_dir / "old_curve.csv"
    stale_file.write_text("x,y\n0,0\n", encoding="utf-8")

    result = ExtractionResult(
        folder_name="PU_TEST",
        curves=[
            CurveData(
                sample_id="SampleA",
                curve_type="stress_strain",
                x=[0.0, 1.0, 2.0],
                y=[0.0, 2.0, 3.0],
            )
        ],
    )

    write_results(output_dir, result)

    assert not stale_file.exists()
    assert (stale_curve_dir / "SampleA_stress_strain.csv").exists()


def test_classify_pdf_si_requires_standalone_token():
    assert classify_pdf(Path("synthesis_route.pdf")) == "main"
    assert classify_pdf(Path("physical_properties.pdf")) == "main"
    assert classify_pdf(Path("paper_SI.pdf")) == "si"
    assert classify_pdf(Path("supporting_information.pdf")) == "si"


def test_load_config_resolves_root_dir_relative_to_config_file(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "agent_config.yaml"
    config_path.write_text("root_dir: papers\nfolders: [PU_001]\n", encoding="utf-8")

    config = load_config(str(config_path), env_path=str(tmp_path / "missing.env"))

    assert config.root_dir == config_dir / "papers"
    assert config.folders == ["PU_001"]


def test_write_results_keeps_multiple_curves_for_same_sample_and_type(tmp_path: Path):
    output_dir = tmp_path / "agent_output"
    result = ExtractionResult(
        folder_name="PU_TEST",
        curves=[
            CurveData(
                sample_id="SampleA",
                curve_type="stress_strain",
                x=[0.0, 1.0],
                y=[0.0, 2.0],
                source_figure="Figure 1",
                source_page=3,
                label="run1",
            ),
            CurveData(
                sample_id="SampleA",
                curve_type="stress_strain",
                x=[0.0, 1.0],
                y=[0.0, 3.0],
                source_figure="Figure S2",
                source_page=7,
                label="run2",
            ),
        ],
    )

    write_results(output_dir, result)

    curve_files = sorted((output_dir / "curves").glob("*.csv"))
    assert len(curve_files) == 2
    assert {path.stem for path in curve_files} == {
        "SampleA_stress_strain_Figure_1_p3_run1",
        "SampleA_stress_strain_Figure_S2_p7_run2",
    }


def test_axis_model_selects_log_from_tick_fit():
    ticks = [
        {"pixel": 0.0, "value": 1.0},
        {"pixel": 50.0, "value": 10.0},
        {"pixel": 100.0, "value": 100.0},
    ]

    calibration = fit_axis_model(ticks, "y")

    assert calibration["scale_type"] == "log"
    assert calibration["r2_log"] > calibration["r2_linear"]
    assert round(map_pixel_to_value(50.0, calibration), 6) == 10.0


def test_axis_model_detects_reversed_ftir_axis():
    ticks = [
        {"pixel": 0.0, "value": 4000.0},
        {"pixel": 100.0, "value": 400.0},
    ]

    calibration = fit_axis_model(ticks, "x")

    assert calibration["scale_type"] == "linear"
    assert calibration["reversed"] is True
    assert map_pixel_to_value(0.0, calibration) > map_pixel_to_value(100.0, calibration)


def test_curve_metadata_prompt_forbids_llm_xy_data():
    prompt = build_curve_metadata_prompt("stress_strain", "Figure 3a. Stress-strain curves.", ["PU-1"])

    assert "Do not output curve x-y data" in prompt
    assert '"curves"' not in prompt
    assert '"data"' not in prompt


def test_curve_csv_contains_point_provenance(tmp_path: Path):
    output_dir = tmp_path / "agent_output"
    result = ExtractionResult(
        folder_name="PU_TEST",
        curves=[
            CurveData(
                sample_id="SampleA",
                curve_type="stress_strain",
                x=[0.0],
                y=[1.0],
                source_pdf="paper.pdf",
                source_page=3,
                source_figure="Figure 3a",
                caption="Stress-strain curves",
                extraction_method="opencv_raster_digitization",
                confidence="medium",
            )
        ],
    )

    write_results(output_dir, result)

    csv_text = next((output_dir / "curves").glob("*.csv")).read_text(encoding="utf-8")
    assert "source_pdf,source_page,source_figure" in csv_text
    assert "paper.pdf,3,Figure 3a" in csv_text
