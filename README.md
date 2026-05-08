# Surgical Planning AI

AI assistant for **perioperative planning**. It collects and structures the main elements of a surgical case from the operative indication and patient data, so the team can review risks, staging, and logistics with clearer context.

Clinical decisions remain entirely with the care team; the tool supports them with organized information and AI-assisted reasoning, not prescriptions.

## Run locally

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Tests

LLM-based evaluation suite (DeepEval — requires `OPENAI_API_KEY`):

```bash
uv run deepeval test run ./test
```

Deterministic safety tests only (no LLM calls, fast feedback):

```bash
uv run pytest test/test_input_guard_node.py test/test_critic_node.py test/test_safety_guards.py test/test_surgical_planning_input.py -v
```

## Safety architecture

The graph is wrapped by two safety stages around the LLM nodes:

```
START
  -> input_guard_node           # pre-LLM: prompt-injection / scope / PII / length
  -> ASA_classifier_node
  -> perioperative_checklist_node
  -> postoperative_care_node
  -> critic_node                # post-LLM: deterministic clinical rules + output PII/scope
  -> END
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
4. **Deterministic critic** (`graph/nodes/critic_node.py`)
   - Internal-consistency rules (`overall_status` vs `critical_alerts` vs item-level alerts).
   - ASA / postoperative destination conflict (ASA IV-V routed to ward).
   - NSAID safety against contraindicating comorbidities.
   - Re-validates every free-text field produced by the LLM (PII, scope).
5. **Domain exception + audit log**
   - All violations raise `safety.exceptions.GuardrailViolation`.
   - The FastAPI handler in `app/main.py` translates them into HTTP 422
     with a non-leaky public message and writes the structured detail to
     the `safety.audit` logger for compliance review.