from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from graph import state
from graph.schema.perioperative_checklist_output import PerioperativeChecklistOutput


def perioperative_checklist_node(state: state):
    """
    Generates a WHO Surgical Safety Checklist (Sign-In, Time-Out, Sign-Out)
    tailored to the patient's profile across any surgical specialty.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are a senior perioperative safety officer responsible for completing the WHO Surgical Safety Checklist
        for any surgical procedure across all specialties. Your task is to evaluate the patient data below
        and dynamically generate a clinically grounded checklist for each of the three standard phases:
        Sign-In (before anesthesia induction), Time-Out (before skin incision), and Sign-Out (before the patient
        leaves the operating room).

        ## Patient Data
        - Age: {age} years
        - Active comorbidities:
        {comorbidities}
        - ASA Physical Status Classification: {asa}

        ## Instructions
        1. For each phase, generate only the checklist items that are clinically relevant given the patient's
           specific profile (age, comorbidities, ASA class). Do not produce generic or inapplicable items.
        2. Each item must have a concise `item` label (e.g., "Difficult airway assessment") and a `notes` field
           explaining briefly why this item is important in this patient's context.
        3. Set `alert=True` whenever the patient data reveals a specific risk or contraindication for that item
           (e.g., coagulopathy increasing blood loss risk, hemodynamic fragility from ASA ≥ III, allergy
           conflicting with standard antibiotic prophylaxis). When `alert=True`, `notes` must include a concise
           clinical justification and the recommended corrective action or precaution.
        4. For `overall_status`:
           - Use "critical" ONLY if at least one unresolved safety conflict could result in immediate harm
             (typical for ASA IV or V, or unresolved severe uncontrolled disease).
           - Use "hold" when one or more items require team review or verification before proceeding
             (typical for ASA II or III with relevant alerts).
           - Use "clear" when all phases are fully verified and no alerts are present
             (typical for ASA I; or ASA II with fully controlled disease and no item flagged alert=True).
        5. `critical_alerts` MUST be empty for ASA I and ASA II patients with controlled disease.
           Populate `critical_alerts` ONLY when `overall_status` is "critical", with a clear, actionable
           description of each unresolved life-threatening issue.
        6. `recommendations` are non-blocking, informational suggestions and MUST be specific to each
           comorbidity present. Apply the following minimum coverage rules:
           - Hypertension → include perioperative blood pressure monitoring and antihypertensive continuation.
           - Diabetes → include glucose monitoring, glycemic control targets, and wound healing considerations.
           - COPD/Asthma → include pulmonary optimization and bronchodilator availability.
           - Heart failure / ischemic heart disease → include cardiology consult before proceeding when
             severity is moderate or higher, plus advanced hemodynamic monitoring.
           - Chronic kidney disease → include renal function review and avoidance of nephrotoxic agents.
           - Septic shock / multi-organ failure → include ICU-level monitoring, vasopressor readiness, and
             resuscitation plan.
           Each recommendation must be a complete, standalone sentence — never reference critical_alerts as
           a substitute for an explicit recommendation entry.
        7. Pediatric considerations (age < 18 years): the Sign-In MUST include weight-based drug dosing
           verification AND parental/guardian informed consent, and at least one item must address
           pediatric airway considerations (smaller airway, age-appropriate equipment).
        8. Geriatric considerations (age ≥ 65 years): include at least one item addressing fragility,
           polypharmacy review, and risk of postoperative delirium / pressure injury.
        9. Internal consistency rule: if `overall_status` is "clear", then no item may have `alert=True`
           and `critical_alerts` must be empty. If `overall_status` is "critical", then `critical_alerts`
           must be non-empty.

        Be precise, evidence-based, and consistent with current WHO and ACSA perioperative safety guidelines.
        """
    )

    llm = init_chat_model("gpt-4o", temperature=0.0)
    llm = llm.with_structured_output(PerioperativeChecklistOutput)

    chain = prompt | llm

    perioperative_checklist = chain.invoke(
        {
            "age": state["age"],
            "comorbidities": state["comorbidities"],
            "asa": state["asa"].asa
        }
    )

    return {"perioperative_checklist": perioperative_checklist}
