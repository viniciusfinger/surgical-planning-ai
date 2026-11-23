from observability.critic_scores import build_critic_scores
from safety.exceptions import GuardrailViolation


def _by_name(scores):
    return {s["name"]: s for s in scores}


def _v(rule):
    return {"rule": rule, "field": None, "detail": None}


def test_clean_run_emits_zeroed_scores():
    scores = _by_name(build_critic_scores([], 0))
    assert scores["critic_outcome"]["value"] == "passed_first_try"
    assert scores["self_correction_retries"]["value"] == 0
    for name in (
        "critic.r1_checklist_consistency",
        "critic.r2_asa_destination",
        "critic.r3_nsaid_safety",
    ):
        assert scores[name]["value"] == 0
        assert "comment" not in scores[name]
    assert "guardrail_block" not in scores


def test_self_corrected_run_counts_rule_hits():
    history = [
        _v("nsaid_with_contraindication"),
        _v("checklist_clear_with_alert_items"),
        _v("nsaid_with_contraindication"),
    ]
    scores = _by_name(build_critic_scores(history, 2))
    assert scores["critic_outcome"]["value"] == "self_corrected"
    assert scores["self_correction_retries"]["value"] == 2
    assert scores["critic.r3_nsaid_safety"]["value"] == 2
    assert scores["critic.r1_checklist_consistency"]["value"] == 1
    assert scores["critic.r2_asa_destination"]["value"] == 0


def test_max_retries_block():
    exc = GuardrailViolation(rule="max_retries_exceeded", stage="output")
    scores = _by_name(
        build_critic_scores([_v("asa_destination_conflict")] * 3, 2, blocked_by=exc)
    )
    assert scores["critic_outcome"]["value"] == "blocked"
    assert scores["guardrail_block"]["value"] == "output:max_retries_exceeded"
    assert scores["critic.r2_asa_destination"]["value"] == 3


def test_validator_message_never_leaks_into_score():
    exc = GuardrailViolation(
        rule="Found PII: John Doe 123.456.789-00", stage="input", field="comorbidities[0].name"
    )
    scores = _by_name(build_critic_scores([], 0, blocked_by=exc))
    assert scores["guardrail_block"]["value"] == "input:guard_validator"
    assert "John" not in str(scores)
