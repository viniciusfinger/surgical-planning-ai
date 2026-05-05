from pydantic import BaseModel

from api.schema.urgency import Urgency
from datetime import date

class SurgicalPlanningInput(BaseModel):
    birthdate: date
    comorbidity: list[str]
    surgical_type: str
    urgency: Urgency
