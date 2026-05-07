from typing import NotRequired, TypedDict

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency
from graph.schema.ASA_output import ASAOutput


class GraphState(TypedDict):
    age: int
    comorbidities: list[Comorbidity]
    surgical_type: str
    urgency: Urgency
    asa: NotRequired[ASAOutput]