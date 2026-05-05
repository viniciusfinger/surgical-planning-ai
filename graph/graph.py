from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from state import GraphState

def create_graph() -> CompiledStateGraph:
    graph = StateGraph(state_schema=GraphState)

    return graph.compile()

