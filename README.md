# Surgical Planning AI

AI assistant for **perioperative planning**. It collects and structures the main elements of a surgical case from the operative indication and patient data, so the team can review risks, staging, and logistics with clearer context.

Clinical decisions remain entirely with the care team; the tool supports them with organized information and AI-assisted reasoning, not prescriptions.

## Run locally

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## What the API does

The API exposes a single endpoint that orchestrates a multi-step LangGraph
pipeline. Given a minimal patient profile (date of birth, comorbidities,
surgical type, urgency), it returns a structured perioperative plan composed of:

- **`asa`** — ASA Physical Status classification with confidence and
  justification.
- **`perioperative_checklist`** — Surgical Safety Checklist items grouped by
  phase (`sign_in`, `time_out`, `sign_out`), plus an `overall_status`,
  critical alerts and recommendations.
- **`postoperative_care`** — postoperative destination, multimodal analgesia,
  prophylaxis (VTE / SSI / PONV), ERAS items, early mobilization,
  discharge criteria, follow-up plan, and clinical alerts.

The original input fields (`age`, `comorbidities`, `surgical_type`, `urgency`)
are echoed back alongside the AI-generated sections so the response is
self-contained and auditable.

### Example

**Request** — `POST /surgical-planning/`

```json
{
  "birthdate": "1965-03-15",
  "comorbidities": [
    {
      "name": "Systemic arterial hypertension",
      "severity": "moderate",
      "controlled": true
    },
    {
      "name": "Type 2 diabetes mellitus",
      "severity": "mild",
      "controlled": false
    }
  ],
  "surgical_type": "Laparoscopic cholecystectomy",
  "urgency": "elective"
}
```

**Response**

```json
{
  "age": 61,
  "comorbidities": [
    {
      "name": "Systemic arterial hypertension",
      "severity": "moderate",
      "controlled": true
    },
    {
      "name": "Type 2 diabetes mellitus",
      "severity": "mild",
      "controlled": false
    }
  ],
  "surgical_type": "Laparoscopic cholecystectomy",
  "urgency": "elective",
  "asa": {
    "asa": "III",
    "confidence": 0.85,
    "justification": "The patient has two comorbidities: moderate systemic arterial hypertension, which is controlled, and mild type 2 diabetes mellitus, which is uncontrolled. The presence of a moderate condition, even when controlled, typically suggests an ASA II classification. However, the additional presence of an uncontrolled condition, albeit mild, increases the overall systemic burden. Given the combination of these factors, the classification leans towards ASA III, reflecting a more conservative approach due to the potential perioperative impact of the uncontrolled diabetes."
  },
  "perioperative_checklist": {
    "sign_in": [
      {
        "item": "Blood pressure / antihypertensive medication review",
        "alert": false,
        "notes": "Reviewing blood pressure and antihypertensive medication is crucial due to the patient's moderate hypertension, ensuring stability before anesthesia."
      },
      {
        "item": "Blood glucose / glycemic control",
        "alert": true,
        "notes": "Patient has uncontrolled Type 2 diabetes, necessitating careful monitoring and management of blood glucose levels to prevent perioperative complications."
      },
      {
        "item": "Cardiac stability verification",
        "alert": true,
        "notes": "ASA III status and hypertension require verification of cardiac stability to prevent intraoperative hemodynamic instability."
      }
    ],
    "time_out": [
      {
        "item": "Blood glucose / glycemic control",
        "alert": true,
        "notes": "Continued monitoring of blood glucose levels is essential due to the patient's uncontrolled diabetes, reducing the risk of intraoperative complications."
      }
    ],
    "sign_out": [
      {
        "item": "Wound healing considerations",
        "alert": false,
        "notes": "Due to diabetes, ensure proper wound care and monitoring to prevent postoperative complications."
      }
    ],
    "overall_status": "hold",
    "critical_alerts": [],
    "recommendations": [
      "Ensure continuous perioperative blood pressure monitoring and continuation of antihypertensive medications.",
      "Implement glucose monitoring and establish glycemic control targets to manage the patient's diabetes effectively.",
      "Consider wound healing implications due to diabetes and plan for enhanced postoperative care."
    ]
  },
  "postoperative_care": {
    "destination": "PACU",
    "destination_rationale": "The patient is ASA III with moderate systemic arterial hypertension and uncontrolled type 2 diabetes mellitus. Although the laparoscopic cholecystectomy is an elective and minimally invasive procedure, the comorbidities warrant close monitoring in the PACU for potential hemodynamic instability and glycemic control.",
    "analgesia_recommendation": [
      {
        "agent": "Paracetamol",
        "route": "PO",
        "dose_or_regimen": "1g q6h",
        "who_step": "step_1",
        "notes": "First-line non-opioid analgesic."
      },
      {
        "agent": "Dexamethasone",
        "route": "IV",
        "dose_or_regimen": "8mg once",
        "who_step": "step_1",
        "notes": "Anti-inflammatory and antiemetic properties."
      },
      {
        "agent": "Tramadol",
        "route": "PO",
        "dose_or_regimen": "50mg q8h PRN",
        "who_step": "step_2",
        "notes": "Weak opioid added for breakthrough pain if non-opioids are insufficient."
      }
    ],
    "prophylaxis_recommendation": [
      {
        "target": "TEV",
        "intervention": "LMWH (Enoxaparin) 40mg SC daily",
        "alert": false,
        "notes": "Moderate risk due to age and ASA class; no contraindications to anticoagulation."
      },
      {
        "target": "IRAS",
        "intervention": "Single dose of Cefazolin 2g IV pre-op",
        "alert": false,
        "notes": "Standard prophylaxis for laparoscopic cholecystectomy."
      },
      {
        "target": "NVPO",
        "intervention": "Ondansetron 4mg IV q8h PRN",
        "alert": false,
        "notes": "Moderate risk due to potential opioid use; ondansetron for prophylaxis."
      }
    ],
    "eras_recommendations": [
      "Early oral fluid intake within 2 hours post-op.",
      "Opioid-sparing analgesia strategy.",
      "Glycemic control with regular monitoring and insulin adjustment.",
      "Minimization of drains and catheters."
    ],
    "early_mobilization": [
      "Begin mobilization 6 hours post-op with sitting and dangling.",
      "Encourage ambulation within 24 hours post-op.",
      "Respiratory physiotherapy due to ASA III status."
    ],
    "discharge_criteria": [
      {
        "scale": "Aldrete",
        "minimum_score": 9,
        "specific_criteria": [
          "Pain controlled with oral analgesics.",
          "Stable vital signs.",
          "Adequate oxygenation without supplemental oxygen."
        ]
      },
      {
        "scale": "PADSS",
        "minimum_score": 9,
        "specific_criteria": [
          "Tolerating oral intake.",
          "Ambulating independently.",
          "Pain and nausea controlled."
        ]
      }
    ],
    "follow_up_plan": [
      "Return visit in 7 days for wound assessment and suture removal.",
      "Medication reconciliation focusing on antihypertensives and antidiabetics.",
      "Patient education on signs of infection, diet, and activity restrictions.",
      "Referral to endocrinologist for diabetes management."
    ],
    "critical_alerts": [
      "Monitor for potential glycemic instability due to uncontrolled diabetes."
    ],
    "recommendations": [
      "Encourage a balanced diet with adequate protein for wound healing.",
      "Target fasting blood glucose <140 mg/dL and postprandial <180 mg/dL.",
      "Promote sleep hygiene to aid recovery.",
      "Set rehabilitation goals for gradual increase in physical activity."
    ]
  }
}
```

## Tests & Evaluations

The evaluation suite is split in two layers:

- **Deterministic** (`test/test_input_guard_node.py`, `test_critic_node.py`,
  `test_critic_scores.py`,
  `test_safety_guards.py`, `test_surgical_planning_input.py`) — pure Python,
  no LLM calls.
- **LLM-based** (`test/test_ASA_classification_node.py`,
  `test_perioperative_checklist_node.py`, `test_postoperative_care_node.py`) —
  parametrize over versioned `Golden`s living in `evals/datasets/` and apply
  a mix of deterministic clinical metrics (`evals/metrics/clinical.py`) and
  `GEval` judges (`evals/presets.py`).

### Run as pytest (CI-friendly)

```bash
uv run deepeval test run ./test                          # full suite, sequential
uv run deepeval test run ./test -m asa_exact_match       # filter by metric mark
```

`-n N` (xdist subprocess parallelism) is **not recommended on Windows when
the project lives under a path with non-ASCII characters** (e.g.
`Área de Trabalho`, `Documentos do João`). On those setups xdist crashes
in `execnet` with `UnicodeEncodeError: ... '\udc81' ... surrogates not
allowed` and reports "No test cases found". Use the batch runner below
instead — it parallelizes inside a single process via asyncio and is
unaffected.

Deterministic tests in isolation (no LLM calls, fast feedback):

```bash
uv run pytest test/test_input_guard_node.py test/test_critic_node.py test/test_critic_scores.py test/test_safety_guards.py test/test_surgical_planning_input.py -v
```

### Run as a batch evaluation (`evaluate()`)

This path uses the same Goldens and metrics as pytest, but invokes
`deepeval.evaluate()` directly. It parallelizes via `asyncio` (no
subprocess spawning), prints a per-category aggregate, and writes a run
record consumable by Confident AI.

```bash
uv run python -m evals.run_asa          # ASA classifier only
uv run python -m evals.run_checklist    # perioperative checklist only
uv run python -m evals.run_postop       # postoperative care only
uv run python -m evals.run_all          # all three back-to-back
```

Tune the parallelism with `--concurrency N` (alias `-c N`) or the
`EVAL_CONCURRENCY` env var (default 8; set to 1 to debug a single golden):

```powershell
uv run python -m evals.run_all --concurrency 16
$env:EVAL_CONCURRENCY = 16; uv run python -m evals.run_all
```

Each script ends with a table like:

```
 Metrics
 -----------------------------------------------------------------------
 metric                                     mean       pass   errors
 ASA Exact Match                            0.92       11/12        0
 ASA Confidence Floor                       1.00       12/12        0
 ASA Clinical Reasoning                     0.89       11/12        0

 Per-category
 -----------------------------------------------------------------------
 category                                   mean       pass
 elderly_no_comorbidities                   0.94        1/1
 mild_uncontrolled_hypothyroidism_boundary  0.78        0/1
 ...
```

### Adding a new evaluation case

1. Append a `Golden(...)` entry to the relevant file in `evals/datasets/`.
   Use a unique `category` slug — it doubles as the pytest test id and as
   the bucket in the per-category aggregate.
2. (Optional) populate `additional_metadata` with deterministic
   expectations (`expected_asa`, `expected_destination_in`,
   `expected_overall_status_in`, `required_keyword_groups`,
   `forbid_step3`, `min_confidence`).
3. Run `uv run python -m evals.run_<node>` to validate locally.

### Windows note

If running the batch scripts on PowerShell raises a `UnicodeEncodeError`
from Rich (DeepEval's UI library), set the terminal to UTF-8 once per
session:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

## Safety architecture

The graph is wrapped by two safety stages around the LLM nodes, with a
self-correction loop between the critic and the LLM nodes:

```
START
  -> input_guard_node                     # pre-LLM: prompt-injection / scope / PII / length
  -> ASA_classifier_node
  -> perioperative_checklist_node  ──┐
  -> postoperative_care_node       ──┤
                                     v
                                  critic_node   # post-LLM: deterministic clinical rules + output PII/scope
                                     |
                      ┌──────────────┴──────────────────────┐
                      │ violations found (retry < MAX_RETRIES)│
                      v                                       v
          perioperative_checklist_node         postoperative_care_node
          (with correction feedback)           (with correction feedback)
                      └──────────────┬──────────────────────┘
                                     v
                                  critic_node
                                     |
                         no violations / MAX_RETRIES hit
                                     v
                                    END / HTTP 422
```

### Layers of defense

1. **Pydantic structural validation** (`api/schema/`, `domain/schema/`)
  - Typed enums, length and date ranges, list size caps, comorbidity uniqueness.
2. **Pre-LLM guard** (`graph/nodes/input_guard_node.py`)
  - Validates every free-text field that will be interpolated in a prompt
   (`comorbidity.name`, `surgical_type`).
  - Backed by [Guardrails AI](https://github.com/guardrails-ai/guardrails)
  with custom validators in `safety/validators.py`:
  `NoPromptInjection`, `ClinicalScopeOnly`, `NoPII`, `LengthBounds`.
3. **Trust-boundary prompts**
  - All three LLM nodes wrap dynamic data inside `<patient_data>...</patient_data>`
   and instruct the model to treat that block as data, never as instructions.
4. **Deterministic critic + self-correction loop** (`graph/nodes/critic_node.py`)
  - Correctable violations (R1–R3) are returned as structured `CorrectionFeedback`
   instead of raising immediately. The graph routes back to the offending LLM
   node(s) with the feedback injected into the prompt (Reflection pattern).
  - **R1** — Internal consistency (`overall_status` vs `critical_alerts` vs item-level alerts) → retries `perioperative_checklist_node`.
  - **R2** — ASA/destination conflict (ASA IV–V routed to ward) → retries `postoperative_care_node`.
  - **R3** — NSAID safety against contraindicating comorbidities → retries `postoperative_care_node`.
  - **R4** — Output text safety (PII / scope re-validation) → **hard fail**, not correctable via retry.
  - After `MAX_RETRIES = 2` failed attempts, any remaining violation raises `GuardrailViolation(rule="max_retries_exceeded")`.
5. **Domain exception + audit log**
  - All violations raise `safety.exceptions.GuardrailViolation`.
  - The FastAPI handler in `app/main.py` translates them into HTTP 422
  with a non-leaky public message and writes the structured detail to
  the `safety.audit` logger for compliance review.

## Observability (Langfuse)

Every request to `POST /surgical-planning/` produces one Langfuse trace named
`surgical-planning`. The LangGraph run (every node and LLM call) is nested
under it via the LangChain `CallbackHandler`, and the critic's verdict is
attached to the trace as **scores**.

### Setup

Set the Langfuse credentials in `.env` (see `.env.example`). The base URL
must match the region of your Langfuse project; a mismatch returns
`401 Unauthorized` when spans are exported.

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or https://cloud.langfuse.com (EU)
```

### Critic scores

Scores are built in `observability/critic_scores.py` from the critic's
`violation_history` (every violation found across all self-correction
passes) and pushed to the trace by the controller. The first five scores
are emitted on **every** request, with `0` when nothing fired, so rates
and averages can be computed directly in Langfuse.

| Score | Type | Values | Meaning |
|---|---|---|---|
| `critic_outcome` | Categorical | `passed_first_try`, `self_corrected`, `blocked` | Final result of the request. `self_corrected` means the critic found a correctable violation (R1–R3) and a retry fixed it. `blocked` means the request ended in HTTP 422. |
| `self_correction_retries` | Numeric | `0` to `MAX_RETRIES` (2) | How many correction rounds the graph ran before finishing. |
| `critic.r1_checklist_consistency` | Numeric | `0..n` | How many times R1 fired across all critic passes. The comment lists the sub-rules, e.g. `checklist_clear_with_critical_alerts x1`. |
| `critic.r2_asa_destination` | Numeric | `0..n` | Same, for R2 (ASA IV–V routed to `ward`). |
| `critic.r3_nsaid_safety` | Numeric | `0..n` | Same, for R3 (NSAID despite a contraindicating comorbidity). |
| `guardrail_block` | Categorical | `<stage>:<rule>` | **Only on blocked requests.** Says which safety layer stopped it (see below). The comment holds the offending field path. |

`guardrail_block` values:

| Value | Cause |
|---|---|
| `input:guard_validator` | The pre-LLM guard rejected a free-text input field (prompt injection, out of scope, PII, length). |
| `output:guard_validator` | R4: a text generated by the LLM failed the output PII/scope re-validation. |
| `output:max_retries_exceeded` | R1–R3 were still violated after `MAX_RETRIES` correction attempts. |

### Reading the scores

| `critic_outcome` | `self_correction_retries` | R1–R3 counts | What happened |
|---|---|---|---|
| `passed_first_try` | 0 | all 0 | The LLM output satisfied every rule on the first pass. |
| `self_corrected` | 1 | e.g. R3 = 1 | The first output gave an NSAID to a CKD patient; the critic sent feedback and the retry fixed it. |
| `blocked` | 2 | e.g. R3 = 3 | The rule fired on every pass (initial + 2 retries), so the plan was blocked. |
| `blocked` | 0 | all 0 | Blocked by the input guard or by R4. Check `guardrail_block`. |

Questions these scores answer in the Langfuse UI:

- **Self-correction rate:** share of traces with `critic_outcome = self_corrected`.
- **Which rule the model breaks most:** compare the averages of the
  `critic.r*` scores across traces.
- **Is a prompt change helping?** Filter by date or release and compare the
  R1–R3 averages and the `blocked` share before and after.
- **Debugging a violation:** filter traces by `critic.r3_nsaid_safety > 0`
  and open the tree to see the original output, the feedback the critic
  sent and the corrected retry.

### Privacy

Score values and comments only carry rule ids, stage names and field
paths. Validator messages are never used as score values, because they may
echo the rejected content (e.g. a name caught by the PII check).

## TODO List

- [ ] Discovery + Implement deterministic tools/functions for clinical scores and calculations (Caprini, Apfel, Lee, ...);
- [ ] Test + Implement self-consistency with multiple samples in ASA classifier;
- [ ] Add human-in-the-loop (HITL) routing policy for low confidence, ASA >= III, urgency, pediatrics, or critic conflict;
- [ ] Discovery + Implement RAG with mandatory citation and runtime verification of recommendations;
