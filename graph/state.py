from typing import NotRequired, TypedDict

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency
from graph.schema.ASA_output import ASAOutput
from graph.schema.perioperative_checklist_output import PerioperativeChecklistOutput


class GraphState(TypedDict):
    age: int
    comorbidities: list[Comorbidity]
    surgical_type: str
    urgency: Urgency
    asa: NotRequired[ASAOutput]
    perioperative_checklist: NotRequired[PerioperativeChecklistOutput]
