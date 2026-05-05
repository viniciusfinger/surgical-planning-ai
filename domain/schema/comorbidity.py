from pydantic import BaseModel


class Comorbidity(BaseModel):
    name: str
    severity: str #TODO: create enum for mild / moderate / severe
    controlled: bool