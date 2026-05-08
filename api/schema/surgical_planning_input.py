from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency

MAX_COMORBIDITIES = 30
MIN_BIRTHDATE = date(1900, 1, 1)
MAX_PLAUSIBLE_AGE_YEARS = 130


class SurgicalPlanningInput(BaseModel):
    birthdate: date
    comorbidities: list[Comorbidity] = Field(max_length=MAX_COMORBIDITIES)
    surgical_type: str = Field(min_length=2, max_length=200)
    urgency: Urgency

    @field_validator("surgical_type")
    @classmethod
    def _trim_surgical_type(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("surgical_type must have at least 2 characters")
        return stripped

    @field_validator("birthdate")
    @classmethod
    def _validate_birthdate(cls, value: date) -> date:
        today = date.today()
        if value > today:
            raise ValueError("birthdate cannot be in the future")
        if value < MIN_BIRTHDATE:
            raise ValueError("birthdate is implausibly old")
        if (today.year - value.year) > MAX_PLAUSIBLE_AGE_YEARS:
            raise ValueError("birthdate yields an implausible age")
        return value

    @model_validator(mode="after")
    def _check_comorbidity_uniqueness(self) -> "SurgicalPlanningInput":
        names = [c.name.strip().lower() for c in self.comorbidities]
        if len(names) != len(set(names)):
            raise ValueError("duplicate comorbidities detected")
        return self
