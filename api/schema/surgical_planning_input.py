from pydantic import BaseModel

from datetime import date

from domain.schema.urgency import Urgency

class SurgicalPlanningInput(BaseModel):
    birthdate: date
    comorbidity: list[str]
    surgical_type: str
    urgency: Urgency
