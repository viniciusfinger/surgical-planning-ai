from fastapi import APIRouter
from schema.surgical_planning_input import SurgicalPlanningInput

router = APIRouter(prefix="surgical-planning")


router.post("/")
def create(surgical_planning_input: SurgicalPlanningInput):
    return SurgicalPlanningInput

