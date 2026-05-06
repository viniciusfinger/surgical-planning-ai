# Surgical Planning AI

AI assistant for **perioperative planning**. It collects and structures the main elements of a surgical case from the operative indication and patient data, so the team can review risks, staging, and logistics with clearer context.

Clinical decisions remain entirely with the care team; the tool supports them with organized information and AI-assisted reasoning, not prescriptions.

## Run locally

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Tests

```bash
uv run deepeval test run ./test
```
