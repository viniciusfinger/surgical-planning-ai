from pydantic import BaseModel

class SurgicalPlanningInput(BaseModel):
    age: int #TODO: Age or birthdate?
    comorbidity: list[str]
    surgical_type: str
    urgency: str #TODO: create enum?
