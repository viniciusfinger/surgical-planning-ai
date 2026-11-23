from datetime import date

from fastapi import APIRouter
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from api.schema.surgical_planning_input import SurgicalPlanningInput
from graph.graph import get_graph
from observability.critic_scores import build_critic_scores
from safety.exceptions import GuardrailViolation

router = APIRouter(prefix="/surgical-planning")


@router.post("/")
async def create(surgical_planning_input: SurgicalPlanningInput):
    graph = get_graph()

    today = date.today()
    age = today.year - surgical_planning_input.birthdate.year
    if (today.month, today.day) < (surgical_planning_input.birthdate.month, surgical_planning_input.birthdate.day):
        age -= 1

    state = {
        "messages": [],
        "age": age,
        "comorbidities": surgical_planning_input.comorbidities,
        "surgical_type": surgical_planning_input.surgical_type,
        "urgency": surgical_planning_input.urgency,
    }

    with get_client().start_as_current_observation(name="surgical-planning") as span:
        langfuse_handler = CallbackHandler()
        try:
            result = await graph.ainvoke(state, config={"callbacks": [langfuse_handler]})
        except GuardrailViolation as exc:
            scores = build_critic_scores(
                exc.metadata.get("violation_history", []),
                exc.metadata.get("retry_count", 0),
                blocked_by=exc,
            )
            _push_scores(span, scores)
            raise

        _push_scores(
            span,
            build_critic_scores(
                result.get("violation_history", []),
                result.get("retry_count", 0),
            ),
        )

    return result


def _push_scores(span, scores: list[dict]) -> None:
    for score in scores:
        span.score_trace(**score)
