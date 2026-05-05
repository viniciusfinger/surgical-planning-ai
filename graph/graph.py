from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes.risk_classifier_node import risk_classifier_node
from graph.state import GraphState


_compiled: CompiledStateGraph | None = None


def _create_graph() -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("risk_classifier_node", risk_classifier_node)
    graph.add_edge(START, "risk_classifier_node")
    graph.add_edge("risk_classifier_node", END)
    return graph.compile()


def get_graph() -> CompiledStateGraph:
    global _compiled
    if _compiled is None:
        _compiled = _create_graph()
    return _compiled
