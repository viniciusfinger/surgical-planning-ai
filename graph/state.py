from typing import NotRequired, TypedDict

from domain.schema.comorbidity import Comorbidity
from domain.schema.urgency import Urgency
from graph.schema.ASA_output import ASAOutput
from graph.schema.perioperative_checklist_output import PerioperativeChecklistOutput
from graph.schema.postoperative_care_output import PostoperativeCareOutput


class CorrectionFeedback(TypedDict):
    rule: str
    field: str | None
    detail: str | None


class GraphState(TypedDict):
    age: int
    comorbidities: list[Comorbidity]
    surgical_type: str
    urgency: Urgency
    asa: NotRequired[ASAOutput]
    perioperative_checklist: NotRequired[PerioperativeChecklistOutput]
    postoperative_care: NotRequired[PostoperativeCareOutput]
    checklist_feedback: NotRequired[list[CorrectionFeedback]]
    postop_feedback: NotRequired[list[CorrectionFeedback]]
    retry_count: NotRequired[int]
