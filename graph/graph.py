from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes.ASA_classifier_node import ASA_classifier_node
from graph.nodes.perioperative_checklist_node import perioperative_checklist_node
from graph.nodes.postoperative_care_node import postoperative_care_node
from graph.state import GraphState


_compiled: CompiledStateGraph | None = None


def _create_graph() -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("ASA_classifier_node", ASA_classifier_node)
    graph.add_node("perioperative_checklist_node", perioperative_checklist_node)
    graph.add_node("postoperative_care_node", postoperative_care_node)
    graph.add_edge(START, "ASA_classifier_node")
    graph.add_edge("ASA_classifier_node", "perioperative_checklist_node")
    graph.add_edge("perioperative_checklist_node", "postoperative_care_node")
    graph.add_edge("postoperative_care_node", END)
    return graph.compile()


def get_graph() -> CompiledStateGraph:
    global _compiled
    if _compiled is None:
        _compiled = _create_graph()
    return _compiled
