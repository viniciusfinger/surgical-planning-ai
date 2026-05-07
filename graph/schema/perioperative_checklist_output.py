from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChecklistItem(BaseModel):
    """
    Represents a single item in a perioperative safety checklist phase. 
    Each item corresponds to a discrete safety verification step.
    """

    item: str = Field(description="Short name or label of the checklist item.")
    alert: bool = Field(
        default=False,
        description="True when this item requires special attention or poses a risk.",
    )
    notes: Optional[str] = Field(
        default=None,
        description=(
            "Brief note on the clinical relevance of this item given the patient context. "
            "Required when alert=True; recommended for all items."
        ),
    )


class PerioperativeChecklistOutput(BaseModel):
    """
    Models a three-phase surgical safety checklist dynamically tailored to the patient's profile.

    Phases:
        - Sign-in: Checks completed before induction of anesthesia
        - Time-out: Final team pause immediately before skin incision
        - Sign-out: Verification steps before the patient leaves the Operating Room
    """

    sign_in: list[ChecklistItem] = Field(
        description=(
            "Phase 1 — Before induction of anesthesia. "
            "Dynamically generated items relevant to ensuring patient safety before sedation, "
            "based on the patient's age, comorbidities, and ASA class."
        )
    )
    time_out: list[ChecklistItem] = Field(
        description=(
            "Phase 2 — Before skin incision. "
            "Dynamically generated items for the final team pause before cutting, "
            "tailored to the specific procedure and patient risk profile."
        )
    )
    sign_out: list[ChecklistItem] = Field(
        description=(
            "Phase 3 — Before patient leaves the operating room. "
            "Dynamically generated items to confirm completeness and safe handover, "
            "considering the procedure performed and patient condition."
        )
    )
    overall_status: Literal["clear", "hold", "critical"] = Field(
        description=(
            "Aggregated safety status across all three phases. "
            "'clear' = all items verified with no alerts. "
            "'hold' = one or more items flagged, requires review before proceeding. "
            "'critical' = one or more critical alerts present, procedure must not proceed."
        )
    )
    critical_alerts: list[str] = Field(
        default_factory=list,
        description=(
            "List of critical safety issues that must be resolved before proceeding. "
            "Populated by cross-referencing checklist items with ASA output and patient data."
        ),
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Non-blocking recommendations to improve safety or care quality.",
    )
