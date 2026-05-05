from typing import Annotated, TypedDict
from langchain.messages import AnyMessage
import operator

from domain.schema.urgency import Urgency

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    age: int
    comorbidity: list[str]
    surgical_type: str
    urgency: Urgency