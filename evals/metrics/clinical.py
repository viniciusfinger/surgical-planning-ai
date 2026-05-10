"""Deterministic, domain-specific DeepEval metrics.

Every metric here is a pure function of the structured Pydantic payload
produced by an LLM node, which the runner stashes in
``LLMTestCase.additional_metadata`` under a well-known key. The metrics
mirror the hard rules already encoded in :mod:`graph.nodes.critic_node`,
exposing them as numeric scores so the team can track regressions across
runs and aggregate per-category KPIs.

Each metric returns a 0/1 score by default (deterministic check), which
plays well with DeepEval's aggregation and the Confident AI dashboard.
Subclassing keeps the boilerplate (`__name__`, `is_successful`,
`a_measure`) in one place.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from graph.schema.ASA_output import ASAOutput
from graph.schema.perioperative_checklist_output import PerioperativeChecklistOutput
from graph.schema.postoperative_care_output import PostoperativeCareOutput

ASA_OUTPUT_KEY = "asa_output"
CHECKLIST_OUTPUT_KEY = "checklist_output"
POSTOP_OUTPUT_KEY = "postop_output"


# Specific NSAID drug names. Word-boundary matched so "aspirin" doesn't fire
# on "aspiration" and Portuguese suffixes (ibuprofen[oa]) are still caught.
# These tokens are screened in *both* ``agent`` and ``notes`` because the
# critic_node already treats notes as an attempt to sneak in a drug.
NSAID_DRUG_PATTERN = re.compile(
    r"\b("
    r"ibuprofen[oa]?|ketorolac[oa]?|cetorolaco|diclofenac[oa]?|"
    r"naproxen[oa]?|celecoxib|etoricoxib|parecoxib|aspirin|"
    r"acetilsalic[íi]lic|aas"
    r")\b",
    re.IGNORECASE,
)

# Class-level mentions ("NSAIDs", "AINEs"). Only screened in ``agent`` —
# notes routinely contain phrases like "avoid NSAIDs due to CKD" which the
# old token-based rule wrongly flagged as a prescription.
NSAID_CLASS_PATTERN = re.compile(r"\b(nsaids?|aines?)\b", re.IGNORECASE)

NSAID_CONTRAINDICATION_TOKENS: tuple[str, ...] = (
    "kidney",
    "renal",
    "ckd",
    "irc",
    "nephropathy",
    "nefropatia",
    "coagulopath",
    "coagulopat",
    "thrombocytopenia",
    "trombocitopenia",
    "heart failure",
    "insuficiência cardíaca",
    "insuficiencia cardiaca",
)


class _DeterministicBaseMetric(BaseMetric):
    """Shared boilerplate for the deterministic metrics defined here."""

    metric_name: str = "Deterministic Metric"

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.metric_name

    @staticmethod
    def _meta(test_case: LLMTestCase) -> dict[str, Any]:
        return test_case.additional_metadata or {}

    def _set(self, score: float, reason: str) -> float:
        self.score = score
        self.success = score >= self.threshold
        self.reason = reason
        return self.score


def _flatten(values: Iterable[str]) -> str:
    return " ".join(value for value in values if value).lower()


class ASAExactMatchMetric(_DeterministicBaseMetric):
    """Validates the predicted ASA class against the golden expectation.

    Two metadata shapes are supported:

    * ``expected_asa`` (str) — strict exact match.
    * ``expected_asa_in`` (list[str]) — accept any ASA in the whitelist.
      Useful for clinically defensible boundary cases (e.g., septic shock
      with multi-organ failure where IV and V are both reasonable).

    When both are provided, ``expected_asa_in`` wins.
    """

    metric_name = "ASA Exact Match"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        asa: ASAOutput | None = meta.get(ASA_OUTPUT_KEY)
        if asa is None:
            return self._set(0.0, "missing asa output")

        whitelist = meta.get("expected_asa_in")
        if whitelist:
            allowed = list(whitelist)
            if asa.asa in allowed:
                return self._set(
                    1.0, f"ASA {asa.asa} is in accepted set {allowed}"
                )
            return self._set(
                0.0,
                f"ASA {asa.asa} not in accepted set {allowed}",
            )

        expected: str | None = meta.get("expected_asa")
        if expected is None:
            return self._set(
                0.0, "missing expected_asa or expected_asa_in metadata"
            )
        if asa.asa == expected:
            return self._set(1.0, f"ASA {asa.asa} matches expected {expected}")
        return self._set(
            0.0,
            f"ASA mismatch — predicted {asa.asa}, expected {expected}",
        )


class ASAConfidenceMinMetric(_DeterministicBaseMetric):
    """1.0 iff ``asa.confidence`` is at least ``min_confidence`` (default 0.85)."""

    metric_name = "ASA Confidence Floor"

    def __init__(self, threshold: float = 1.0, default_minimum: float = 0.85):
        super().__init__(threshold=threshold)
        self.default_minimum = default_minimum

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        asa: ASAOutput | None = meta.get(ASA_OUTPUT_KEY)
        minimum: float = float(meta.get("min_confidence", self.default_minimum))
        if asa is None:
            return self._set(0.0, "missing asa output")
        if asa.confidence >= minimum:
            return self._set(1.0, f"confidence {asa.confidence:.2f} ≥ {minimum:.2f}")
        return self._set(
            0.0,
            f"confidence {asa.confidence:.2f} below minimum {minimum:.2f}",
        )


class DestinationSafetyMetric(_DeterministicBaseMetric):
    """Validates the postoperative destination against safety rules.

    * If ``expected_destination_in`` is provided, the actual destination must
      belong to that whitelist.
    * Independent of the metadata, ASA IV/V routed to ``ward`` always fails
      (mirrors :func:`graph.nodes.critic_node._check_destination_conflict`).
    """

    metric_name = "Destination Safety"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        postop: PostoperativeCareOutput | None = meta.get(POSTOP_OUTPUT_KEY)
        asa: str | None = meta.get("expected_asa")
        if postop is None:
            return self._set(0.0, "missing postop output")

        destination = postop.destination
        if asa in {"IV", "V"} and destination == "ward":
            return self._set(
                0.0,
                f"ASA {asa} patient routed to 'ward' is unsafe",
            )

        whitelist = meta.get("expected_destination_in")
        if whitelist:
            allowed = set(whitelist)
            if destination not in allowed:
                return self._set(
                    0.0,
                    f"destination '{destination}' not in expected {sorted(allowed)}",
                )

        return self._set(1.0, f"destination '{destination}' is safe and expected")


class ProphylaxisCoverageMetric(_DeterministicBaseMetric):
    """Verifies the prophylaxis section covers all required targets.

    By default the required set is ``{"TEV", "IRAS", "NVPO"}``. A golden may
    narrow it via ``additional_metadata['expected_prophylaxis_targets']``
    when omitting one of the targets is clinically defensible (e.g. minor
    pediatric outpatient surgery where TEV prophylaxis is not indicated).
    """

    metric_name = "Prophylaxis Coverage (TEV/IRAS/NVPO)"

    DEFAULT_REQUIRED_TARGETS = frozenset({"TEV", "IRAS", "NVPO"})

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        postop: PostoperativeCareOutput | None = meta.get(POSTOP_OUTPUT_KEY)
        if postop is None:
            return self._set(0.0, "missing postop output")

        required = frozenset(
            meta.get("expected_prophylaxis_targets")
            or self.DEFAULT_REQUIRED_TARGETS
        )
        targets = {p.target for p in postop.prophylaxis_recommendation}
        missing = required - targets
        if missing:
            return self._set(
                0.0,
                f"missing prophylaxis targets: {sorted(missing)}",
            )

        for item in postop.prophylaxis_recommendation:
            if not item.intervention.strip():
                return self._set(0.0, f"empty intervention for target {item.target}")
            if item.alert and not (item.notes or "").strip():
                return self._set(
                    0.0,
                    f"target {item.target} has alert=True without notes",
                )

        return self._set(
            1.0,
            f"{sorted(required)} covered with non-empty interventions",
        )


class NsaidContraindicationMetric(_DeterministicBaseMetric):
    """Detects NSAIDs prescribed despite contraindicating comorbidities.

    The detection is done in two layers:

    * Specific drug names (``ibuprofen``, ``ketorolac``, ``diclofenac``,
      ``aspirin``…) are word-boundary matched in both the agent and the
      notes — mirroring the critic_node which catches "Add ibuprofen if
      pain persists" hidden in a notes field.
    * Class-level mentions (``NSAIDs``, ``AINEs``) are matched only in the
      agent, never in the notes. This avoids the false positive where the
      LLM correctly writes "avoid NSAIDs due to coagulopathy" inside the
      notes of a non-NSAID drug like dexamethasone.
    """

    metric_name = "NSAID Contraindication Safety"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        postop: PostoperativeCareOutput | None = meta.get(POSTOP_OUTPUT_KEY)
        comorbidities = meta.get("comorbidity_names", [])
        if postop is None:
            return self._set(0.0, "missing postop output")

        comorbidity_blob = " ".join(name.lower() for name in comorbidities)
        has_contraindication = any(
            token in comorbidity_blob for token in NSAID_CONTRAINDICATION_TOKENS
        )
        if not has_contraindication:
            return self._set(1.0, "no NSAID-contraindicating comorbidity present")

        offenders: list[str] = []
        for protocol in postop.analgesia_recommendation:
            agent = protocol.agent or ""
            notes = protocol.notes or ""
            agent_haystack = f"{agent} {notes}"
            if NSAID_DRUG_PATTERN.search(agent_haystack):
                offenders.append(agent)
                continue
            if NSAID_CLASS_PATTERN.search(agent):
                offenders.append(agent)

        if offenders:
            return self._set(
                0.0,
                "NSAID prescribed despite contraindicating comorbidity: "
                f"{offenders}",
            )
        return self._set(1.0, "analgesia plan respects NSAID contraindication")


class WHOLadderMetric(_DeterministicBaseMetric):
    """Enforces WHO analgesic ladder defaults.

    * Always at least one ``step_1`` (non-opioid) agent.
    * If ``forbid_step3`` is True in the metadata, no ``step_3`` agent.
    """

    metric_name = "WHO Analgesic Ladder"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        postop: PostoperativeCareOutput | None = meta.get(POSTOP_OUTPUT_KEY)
        if postop is None:
            return self._set(0.0, "missing postop output")

        steps = {a.who_step for a in postop.analgesia_recommendation}
        if "step_1" not in steps:
            return self._set(0.0, "no WHO step_1 (non-opioid) agent present")

        if meta.get("forbid_step3") and "step_3" in steps:
            return self._set(
                0.0,
                "step_3 (strong opioid) prescribed in a context that forbids it",
            )

        return self._set(1.0, "analgesia plan respects WHO ladder constraints")


class OverallStatusMetric(_DeterministicBaseMetric):
    """Checks that ``perioperative_checklist.overall_status`` is in the
    ``expected_overall_status_in`` whitelist (if provided)."""

    metric_name = "Checklist Overall Status"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        checklist: PerioperativeChecklistOutput | None = meta.get(CHECKLIST_OUTPUT_KEY)
        whitelist = meta.get("expected_overall_status_in")
        if checklist is None:
            return self._set(0.0, "missing checklist output")
        if whitelist is None:
            return self._set(1.0, "no expected status whitelist; skipping check")

        allowed = set(whitelist)
        if checklist.overall_status in allowed:
            return self._set(
                1.0,
                f"overall_status '{checklist.overall_status}' is allowed",
            )
        return self._set(
            0.0,
            f"overall_status '{checklist.overall_status}' not in {sorted(allowed)}",
        )


class PhaseCompletenessMetric(_DeterministicBaseMetric):
    """All three checklist phases (Sign-In, Time-Out, Sign-Out) must be
    populated with at least one item. This was an implicit assertion in the
    previous pytest suite (``assert len(checklist.time_out) > 0``); promoting
    it to a metric makes the failure mode explicit in the aggregate report
    instead of relying on the GEval judge to surface the empty-phase issue.
    """

    metric_name = "Checklist Phase Completeness"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        checklist: PerioperativeChecklistOutput | None = meta.get(CHECKLIST_OUTPUT_KEY)
        if checklist is None:
            return self._set(0.0, "missing checklist output")

        empty_phases: list[str] = []
        if not checklist.sign_in:
            empty_phases.append("sign_in")
        if not checklist.time_out:
            empty_phases.append("time_out")
        if not checklist.sign_out:
            empty_phases.append("sign_out")

        if empty_phases:
            return self._set(
                0.0,
                f"empty checklist phase(s): {empty_phases}",
            )
        return self._set(
            1.0,
            "all three phases (sign_in / time_out / sign_out) populated",
        )


class AlertNotesCompletenessMetric(_DeterministicBaseMetric):
    """Every checklist item with ``alert=True`` must carry non-empty notes."""

    metric_name = "Alert Notes Completeness"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        checklist: PerioperativeChecklistOutput | None = meta.get(CHECKLIST_OUTPUT_KEY)
        if checklist is None:
            return self._set(0.0, "missing checklist output")

        offenders: list[str] = []
        for phase_name, phase in (
            ("sign_in", checklist.sign_in),
            ("time_out", checklist.time_out),
            ("sign_out", checklist.sign_out),
        ):
            for item in phase:
                if item.alert and not (item.notes or "").strip():
                    offenders.append(f"{phase_name}:{item.item}")

        if offenders:
            return self._set(
                0.0,
                f"{len(offenders)} alerted items lack notes: {offenders[:3]}",
            )
        return self._set(1.0, "every alert=True item has non-empty notes")


class KeywordCoverageMetric(_DeterministicBaseMetric):
    """Validates that *any* keyword from each required group appears in the
    rendered output. Useful for clinical themes that can be expressed many
    ways (e.g., ``{"renal", "kidney", "nephro"}`` for CKD coverage).

    Required groups are read from ``additional_metadata['required_keyword_groups']``
    where each group is a list of synonyms; the metric passes only if *every*
    group has at least one synonym present in ``LLMTestCase.actual_output``.
    """

    metric_name = "Required Keyword Coverage"

    def measure(self, test_case: LLMTestCase) -> float:
        meta = self._meta(test_case)
        groups: list[list[str]] = meta.get("required_keyword_groups") or []
        if not groups:
            return self._set(1.0, "no required keyword groups configured")

        haystack = (test_case.actual_output or "").lower()
        missing_groups: list[list[str]] = []
        for group in groups:
            if not any(keyword.lower() in haystack for keyword in group):
                missing_groups.append(group)

        if missing_groups:
            preview = [g[0] for g in missing_groups[:3]]
            return self._set(
                0.0,
                f"missing coverage for {len(missing_groups)} keyword group(s); "
                f"e.g. {preview}",
            )
        return self._set(1.0, f"all {len(groups)} keyword group(s) covered")


__all__ = [
    "ASA_OUTPUT_KEY",
    "CHECKLIST_OUTPUT_KEY",
    "POSTOP_OUTPUT_KEY",
    "AlertNotesCompletenessMetric",
    "ASAConfidenceMinMetric",
    "ASAExactMatchMetric",
    "DestinationSafetyMetric",
    "KeywordCoverageMetric",
    "NsaidContraindicationMetric",
    "OverallStatusMetric",
    "PhaseCompletenessMetric",
    "ProphylaxisCoverageMetric",
    "WHOLadderMetric",
]
