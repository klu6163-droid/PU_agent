# PU Literature Data Extraction Agent

Python agent for extracting structured polyurethane literature data from paper folders. It reads PDFs, extracts text and tables, asks an Anthropic Messages-compatible LLM for structured metadata, digitizes target figures with a vision model, and writes per-paper plus combined CSV/JSON reports.

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

## GitHub hygiene

Do not commit `.env`, generated outputs, or paper PDFs unless you have a clear license and storage plan. The included `.gitignore` excludes those by default.

If a real API key was ever stored in this working tree, rotate it before publishing.
