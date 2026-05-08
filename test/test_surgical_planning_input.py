from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from api.schema.surgical_planning_input import SurgicalPlanningInput
from domain.schema.comorbidity import Comorbidity
from domain.schema.severity import Severity
from domain.schema.urgency import Urgency


def _comorbidity(name: str) -> Comorbidity:
    return Comorbidity(name=name, severity=Severity.mild, controlled=True)


class TestBirthdate:
    def test_rejects_future_birthdate(self):
        """A birthdate in the future is structurally invalid."""
        with pytest.raises(ValidationError):
            SurgicalPlanningInput(
                birthdate=date(2999, 1, 1),
                comorbidities=[],
                surgical_type="Apendicectomia",
                urgency=Urgency.elective,
            )

    def test_rejects_implausible_age(self):
        """A birthdate that yields an implausible age is rejected."""
        with pytest.raises(ValidationError):
            SurgicalPlanningInput(
                birthdate=date(1700, 1, 1),
                comorbidities=[],
                surgical_type="Apendicectomia",
                urgency=Urgency.elective,
            )

    def test_accepts_realistic_birthdate(self):
        """A birthdate that yields a plausible adult age must be accepted."""
        payload = SurgicalPlanningInput(
            birthdate=date(1980, 5, 1),
            comorbidities=[],
            surgical_type="Apendicectomia",
            urgency=Urgency.elective,
        )
        assert payload.birthdate == date(1980, 5, 1)


class TestComorbidities:
    def test_rejects_duplicate_comorbidities(self):
        """Case-insensitive duplicate comorbidity names are rejected."""
        with pytest.raises(ValidationError):
            SurgicalPlanningInput(
                birthdate=date(1980, 5, 1),
                comorbidities=[
                    _comorbidity("Hypertension"),
                    _comorbidity("hypertension  "),
                ],
                surgical_type="Apendicectomia",
                urgency=Urgency.elective,
            )

    def test_rejects_excessive_comorbidities(self):
        """Lists above the configured cap are rejected to bound prompt size."""
        many = [_comorbidity(f"Condition {i}") for i in range(40)]
        with pytest.raises(ValidationError):
            SurgicalPlanningInput(
                birthdate=date(1980, 5, 1),
                comorbidities=many,
                surgical_type="Apendicectomia",
                urgency=Urgency.elective,
            )


class TestSurgicalType:
    def test_trims_whitespace(self):
        """Leading and trailing whitespace is trimmed at the schema level."""
        payload = SurgicalPlanningInput(
            birthdate=date(1980, 5, 1),
            comorbidities=[],
            surgical_type="  Apendicectomia laparoscopica  ",
            urgency=Urgency.elective,
        )
        assert payload.surgical_type == "Apendicectomia laparoscopica"

    def test_rejects_too_short_surgical_type(self):
        """`surgical_type` shorter than the minimum is rejected by the schema."""
        with pytest.raises(ValidationError):
            SurgicalPlanningInput(
                birthdate=date(1980, 5, 1),
                comorbidities=[],
                surgical_type="a",
                urgency=Urgency.elective,
            )
