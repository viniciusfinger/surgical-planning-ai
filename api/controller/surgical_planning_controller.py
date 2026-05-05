from datetime import date

from fastapi import APIRouter

from api.schema.surgical_planning_input import SurgicalPlanningInput
from graph.graph import get_graph

router = APIRouter(prefix="/surgical-planning")


@router.post("/")
def create(surgical_planning_input: SurgicalPlanningInput):
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

    result = graph.invoke(state)

    return result
