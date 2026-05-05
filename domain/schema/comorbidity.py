from pydantic import BaseModel

from domain.schema.severity import Severity


class Comorbidity(BaseModel):
    name: str
    severity: Severity
    controlled: bool