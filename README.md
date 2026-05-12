# PU Literature Data Extraction Agent

Python agent for extracting structured polyurethane literature data from paper folders. It reads PDFs, extracts text and tables, asks an Anthropic Messages-compatible LLM for structured metadata, digitizes target figures with image/vector algorithms, and writes per-paper plus combined CSV/JSON reports.

## Repository layout

- `agent/` - package code for configuration, LLM calls, extractors, and output writers.
- `agent_config.yaml` - default folders and extraction settings.
- `.env.example` - environment variable template. Copy it to `.env` locally and fill in credentials.
- `requirements.txt` - Python dependencies for the agent.
- `PU_*/` - local paper folders. PDFs and generated outputs are ignored by Git by default.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your Anthropic Messages-compatible endpoint:

```dotenv
LLM_BASE_URL=https://your-endpoint.example.com
LLM_API_KEY=your_api_key
LLM_MODEL=your_text_model
LLM_VISION_MODEL=your_vision_model
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=8192
```

`LLM_BASE_URL` should be the API root before `/v1/messages`; the client appends `/v1/messages`.

## Run

Run all configured folders and all extraction steps:

```powershell
python -m agent.cli
```

Run selected folders:

```powershell
python -m agent.cli --folders PU_001 PU_002
```

Run selected steps:

```powershell
python -m agent.cli --folders PU_001 --steps literature mechanical
python -m agent.cli --folders PU_001 --steps curves
```

Outputs are written to each paper folder under `agent_output/`, plus combined tables under `combined_agent_output/`.

## Curve extraction design

The LLM is not allowed to generate final x-y curve data. For target plots such as stress-strain, FTIR, SAXS, DSC, and XRD/WAXS, the curve workflow is:

1. Use text and a vision LLM only for figure metadata: target page, figure ID, caption, plot type, axis names/units, legend labels, and sample associations.
2. Render and crop the target PDF page.
3. Try the vector-path extraction hook first; fall back to OpenCV raster digitization when no usable path is available.
4. Use HSV/LAB color clustering, connected components, long-line removal for axes/gridlines, and per-column skeleton-style reduction to recover curve pixels.
5. Use OCR tick candidates when available, otherwise domain defaults, then fit both linear and log mappings:
   - `value = a * pixel + b`
   - `log10(value) = a * pixel + b`
6. Select the axis scale by R2, support reversed axes such as FTIR 4000 to 400 cm^-1, and map pixels to physical coordinates.
7. Run plot-specific checks, including stress-strain monotonic/nonnegative checks, FTIR range checks, SAXS positivity and d-spacing estimate, and linear-increasing checks for XRD/DSC.

Each extracted curve keeps PDF/page/figure/caption/calibration/method/confidence provenance in the CSV and JSON outputs.

Curve review artifacts are written under each paper's `agent_output/`:

```text
agent_output/
  figures/
    Figure_3a_original.png
    Figure_3a_crop.png
  curves/
    SampleA_stress_strain_Figure_3a_p3_curve_1.csv
  review/
    Figure_3a_curve_mask.png
    Figure_3a_digitized_overlay.png
    Figure_3a_replotted_curve.png
  calibration/
    Figure_3a_axis_calibration.json
```

OCR uses `pytesseract` when the Tesseract executable is available. If OCR is unavailable or too weak, the pipeline still runs with domain fallback ticks and marks confidence/review notes accordingly.

## GitHub hygiene

Do not commit `.env`, generated outputs, or paper PDFs unless you have a clear license and storage plan. The included `.gitignore` excludes those by default.

If a real API key was ever stored in this working tree, rotate it before publishing.
