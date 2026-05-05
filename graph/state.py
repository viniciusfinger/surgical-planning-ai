from typing import TypedDict

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency

class GraphState(TypedDict):
    age: int
    comorbidities: list[Comorbidity]
    surgical_type: str
    urgency: Urgency