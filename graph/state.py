from typing import Annotated, TypedDict
from langchain.messages import AnyMessage
import operator

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]