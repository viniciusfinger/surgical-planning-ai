from pydantic import BaseModel

from datetime import date

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency

class SurgicalPlanningInput(BaseModel):
    birthdate: date
    comorbidities: list[Comorbidity]
    surgical_type: str #TODO: set a better name
    urgency: Urgency
