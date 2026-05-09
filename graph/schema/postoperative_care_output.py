from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnalgesiaProtocol(BaseModel):
    agent: str = Field(description="Drug or intervention name.")
    route: str = Field(description="Administration route (e.g., IV, PO, epidural).")
    dose_or_regimen: str = Field(description="Dose, frequency, or regimen (e.g., 1g q6h).")
    who_step: Literal["step_1", "step_2", "step_3"] = Field(
        description=(
            "WHO analgesic ladder step: "
            "step_1 = non-opioid, step_2 = weak opioid, step_3 = strong opioid."
        )
    )
    notes: Optional[str] = Field(
        default=None,
        description="Clinical justification or patient-specific caveat.",
    )


class ProphylaxisItem(BaseModel):
    target: Literal["TEV", "IRAS", "NVPO"] = Field(
        description="Prophylaxis target: TEV (thromboembolic), IRAS (surgical site infection), NVPO (postoperative nausea/vomiting)."
    )
    intervention: str = Field(description="Recommended prophylactic measure or drug.")
    alert: bool = Field(
        default=False,
        description="True if a patient-specific risk or contraindication was identified.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Rationale, risk factors, or contraindication details. Required when alert=True.",
    )


class DischargeCriteria(BaseModel):
    scale: Literal["Aldrete", "PADSS"] = Field(
        description=(
            "Scale used: Aldrete for PACU discharge, "
            "PADSS (Post-Anesthetic Discharge Scoring System) for home discharge."
        )
    )
    minimum_score: int = Field(description="Minimum acceptable score for discharge (e.g., ≥9 for Aldrete).")
    specific_criteria: list[str] = Field(
        description="Patient-tailored criteria that must be individually met before discharge."
    )


class PostoperativeCareOutput(BaseModel):
    destination: Literal["PACU", "ICU", "ward"] = Field(
        description=(
            "Recommended immediate postoperative destination: "
            "PACU (Post-Anesthesia Care Unit / SRPA), ICU, or general ward."
        )
    )
    destination_rationale: str = Field(
        description="Clinical justification for the recommended destination based on ASA class, procedure, and comorbidities."
    )
    analgesia_recommendation: list[AnalgesiaProtocol] = Field(
        description=(
            "Multimodal analgesic protocol following the WHO ladder and ERAS principles. "
            "Include at least one non-opioid agent; add opioids only when clinically justified."
        )
    )
    prophylaxis_recommendation: list[ProphylaxisItem] = Field(
        description="Prophylaxis items covering TEV, IRAS, and NVPO, tailored to patient risk profile."
    )
    eras_recommendations: list[str] = Field(
        description=(
            "ERAS (Enhanced Recovery After Surgery) protocol items applicable to the specialty and procedure. "
            "Include only evidence-based items relevant to this patient."
        )
    )
    early_mobilization: list[str] = Field(
        description=(
            "Early mobilization and physiotherapy plan: timeline, goals, and restrictions "
            "considering the procedure performed and patient functional status."
        )
    )
    discharge_criteria: list[DischargeCriteria] = Field(
        description="Discharge criteria using Aldrete (PACU) and/or PADSS (home), customized to the patient profile."
    )
    follow_up_plan: list[str] = Field(
        description=(
            "Outpatient follow-up plan: scheduled visits, wound care, medication reconciliation, "
            "specialist referrals, and patient education points."
        )
    )
    critical_alerts: list[str] = Field(
        default_factory=list,
        description="High-priority postoperative warnings requiring immediate clinical attention.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Non-critical suggestions to optimize postoperative recovery and safety.",
    )
