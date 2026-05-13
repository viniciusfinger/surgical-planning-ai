from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes.ASA_classifier_node import ASA_classifier_node
from graph.nodes.critic_node import critic_node
from graph.nodes.input_guard_node import input_guard_node
from graph.nodes.perioperative_checklist_node import perioperative_checklist_node
from graph.nodes.postoperative_care_node import postoperative_care_node
from graph.state import GraphState


_compiled: CompiledStateGraph | None = None


def _route_after_critic(state: GraphState) -> list[str] | str:
    """
    Routes back to the LLM node(s) that produced a correctable violation,
    or ends the graph when the critic finds no issues.
    """
    checklist_feedback = state.get("checklist_feedback", [])
    postop_feedback = state.get("postop_feedback", [])

    if not checklist_feedback and not postop_feedback:
        return END

    targets: list[str] = []
    if checklist_feedback:
        targets.append("perioperative_checklist_node")
    if postop_feedback:
        targets.append("postoperative_care_node")

    return targets[0] if len(targets) == 1 else targets


def _create_graph() -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("input_guard_node", input_guard_node)
    graph.add_node("ASA_classifier_node", ASA_classifier_node)
    graph.add_node("perioperative_checklist_node", perioperative_checklist_node)
    graph.add_node("postoperative_care_node", postoperative_care_node)
    graph.add_node("critic_node", critic_node)

    graph.add_edge(START, "input_guard_node")
    graph.add_edge("input_guard_node", "ASA_classifier_node")
    graph.add_edge("ASA_classifier_node", "perioperative_checklist_node")
    graph.add_edge("ASA_classifier_node", "postoperative_care_node")
    graph.add_edge("perioperative_checklist_node", "critic_node")
    graph.add_edge("postoperative_care_node", "critic_node")
    graph.add_conditional_edges("critic_node", _route_after_critic)
    return graph.compile()


def get_graph() -> CompiledStateGraph:
    global _compiled
    if _compiled is None:
        _compiled = _create_graph()
    return _compiled
