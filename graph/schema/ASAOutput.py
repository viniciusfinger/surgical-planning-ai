from typing import Literal
from pydantic import BaseModel, Field

class ASAOutput(BaseModel):
    asa: Literal["I", "II", "III", "IV", "V"]
    confidence: float = Field(ge=0, le=1)
    justification: str = Field(description="Short clinical reasoning referencing severity and control")