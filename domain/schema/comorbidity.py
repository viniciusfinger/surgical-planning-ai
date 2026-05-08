from pydantic import BaseModel, Field, field_validator

from domain.schema.severity import Severity


class Comorbidity(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    severity: Severity
    controlled: bool

    @field_validator("name")
    @classmethod
    def _trim_name(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("comorbidity name must have at least 2 characters")
        return stripped
