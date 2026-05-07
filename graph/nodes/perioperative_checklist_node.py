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
           - Use "critical" if any item presents an unresolved patient safety conflict that could result in harm.
           - Use "hold" if one or more items require team review or verification before proceeding.
           - Use "clear" only if all phases are fully verified and no alerts are present.
        5. Populate `critical_alerts` with a clear, actionable description of each critical issue found.
        6. Populate `recommendations` with non-blocking but clinically relevant suggestions to optimize
           perioperative safety and recovery, informed by the patient's age, comorbidities, and ASA class.

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
