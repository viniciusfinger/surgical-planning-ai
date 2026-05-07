from typing import Literal
from pydantic import BaseModel, Field

class ASAOutput(BaseModel):
    asa: Literal["I", "II", "III", "IV", "V"]
    confidence: float = Field(ge=0, le=1)
    justification: str = Field(description="Concise clinical justification for the assigned ASA classification. Must explicitly reference comorbidity severity, control status, functional impact, and overall systemic burden when relevant. Avoid generic explanations or repeating the input data.")