from __future__ import annotations

import re
from typing import Annotated, Any, Callable

import requests
from pydantic import BaseModel, Field
from simpleeval import SimpleEval

import datetime as datetime_module
import math
from decimal import Decimal


class PatientSearchParams(BaseModel):
    birthdate: str | None = Field(
        default=None, description="The patient's date of birth in the format YYYY-MM-DD."
    )
    family: str | None = Field(default=None, description="The patient's family (last) name.")
    given: str | None = Field(
        default=None, description="The patient's given name. May include first and middle names."
    )
    identifier: str | None = Field(
        default=None, description="The patient's identifier or Medical Record Number (MRN)."
    )


class ObservationSearchParams(BaseModel):
    code: str = Field(
        description=(
            "A short lab shorthand code only (e.g., 'K' for potassium, 'A1C' for hemoglobin A1C). "
            "Do not provide LOINC codes or any other coding system identifiers."
        )
    )
    patient: str = Field(description="Reference to a patient resource the condition is for.")


class ProcedureSearchParams(BaseModel):
    code: str | None = Field(default=None, description="External CPT codes associated with the procedure.")
    date: str | None = Field(
        default=None,
        description=(
            "Date or period that the procedure was performed, using the FHIR date parameter format "
            "(e.g., 'ge2023-07-01')."
        ),
    )
    patient: str = Field(description="Reference to a patient resource the condition is for.")


class ConditionSearchParams(BaseModel):
    code: str | None = Field(default=None, description="An ICD-10 diagnosis code to filter conditions by.")
    patient: str = Field(description="Reference to a patient resource the condition is for.")


class VitalsSearchParams(BaseModel):
    category: str = Field(description='Use "vital-signs" to search for vitals observations.')
    date: str | None = Field(default=None, description="The date range for when the observation was taken.")
    patient: str = Field(description="Reference to a patient resource the condition is for.")


class MedicationRequestSearchParams(BaseModel):
    category: str | None = Field(
        default=None,
        description=(
            "The category of medication orders to search for. By default all categories are searched.\n\n"
            "Supported categories:\n"
            "Inpatient\n"
            "Outpatient (those administered in the clinic - CAMS)\n"
            "Community (prescriptions)\n"
            "Discharge"
        ),
    )
    date: str | None = Field(
        default=None,
        description=(
            "The medication administration date. This parameter corresponds to the "
            "dosageInstruction.timing.repeat.boundsPeriod element. Medication orders that do not "
            "have start and end dates within the search parameter dates are filtered."
        ),
    )
    patient: str = Field(description="The FHIR patient ID.")


class Coding(BaseModel):
    system: str = Field(description='Coding system such as "http://loinc.org"')
    code: str = Field(description="The actual code")
    display: str = Field(description="Display name")


class CategoryCoding(BaseModel):
    system: str = Field(description='Use "http://hl7.org/fhir/observation-category"')
    code: str = Field(description='Use "vital-signs"')
    display: str = Field(description='Use "Vital Signs"')


class Category(BaseModel):
    coding: list[CategoryCoding]


class CodeText(BaseModel):
    text: str = Field(description="What is being measured.")


class SubjectReference(BaseModel):
    reference: str = Field(description="Format: Patient/{patient_id}")


class VitalsCreateParams(BaseModel):
    resourceType: str = Field(description='Use "Observation" for vitals observations.')
    category: list[Category]
    code: CodeText
    effectiveDateTime: str = Field(description="ISO datetime when the observation was taken.")
    status: str = Field(description='Only a value of "final" is supported.')
    valueString: str = Field(description="Measurement value")
    subject: SubjectReference


class MedicationCodeableConcept(BaseModel):
    coding: list[Coding]
    text: str = Field(description="The order display name of the medication.")


class DoseQuantity(BaseModel):
    value: float = Field(description="The numeric value")
    unit: str = Field(description="Unit for the dose")


class RateQuantity(BaseModel):
    value: float = Field(description="The numeric value")
    unit: str = Field(description="Unit for the rate")


class DoseAndRate(BaseModel):
    doseQuantity: DoseQuantity
    rateQuantity: RateQuantity


class DosageInstruction(BaseModel):
    route: str = Field(description="The medication route.")
    doseAndRate: list[DoseAndRate]


class MedicationRequestCreateParams(BaseModel):
    resourceType: str = Field(description='Use "MedicationRequest" for medication requests.')
    medicationCodeableConcept: MedicationCodeableConcept
    authoredOn: str = Field(description="The date the prescription was written.")
    dosageInstruction: list[DosageInstruction]
    status: str = Field(description='Use "active".')
    intent: str = Field(description='Use "order".')
    subject: SubjectReference


class Code(BaseModel):
    coding: list[Coding]


class Note(BaseModel):
    text: str = Field(description="Free text comment")


class ServiceRequestCreateParams(BaseModel):
    resourceType: str = Field(description='Use "ServiceRequest" for service requests.')
    code: Code
    authoredOn: str = Field(description="The order instant (signed/signed+held).")
    status: str = Field(description='Use "active".')
    intent: str = Field(description='Use "order".')
    priority: str = Field(description='Use "stat".')
    subject: SubjectReference
    note: Note
    occurrenceDateTime: str = Field(description="ISO datetime when the service request should occur.")


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def create_patient_search(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_patient_search(
        *,
        birthdate: Annotated[str | None, "The patient's date of birth in YYYY-MM-DD format."] = None,
        family: Annotated[str | None, "The patient's family (last) name."] = None,
        given: Annotated[str | None, "The patient's given name. May include first and middle names."] = None,
        identifier: Annotated[str | None, "The patient's identifier or Medical Record Number (MRN)."] = None,
    ) -> dict:
        """Search for a patient via the FHIR Patient search API."""
        route = f"{fhir_api_base}Patient"
        args = PatientSearchParams(
            birthdate=birthdate, family=family, given=given, identifier=identifier
        )
        params = args.model_dump(exclude_none=True)
        res = requests.get(route, params=params)
        return res.json()

    return fhir_patient_search


def create_fhir_observation_search(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_observation_search(
        *,
        search_params: Annotated[ObservationSearchParams, "FHIR Observation search parameters (labs)."],
        explanation: Annotated[str, "Explanation for calling this tool."],
    ) -> dict:
        """Observation.Search (Labs) returns component-level lab result data."""
        route = f"{fhir_api_base}Observation"
        search_params_obj = ObservationSearchParams.model_validate(search_params)
        params = {
            **search_params_obj.model_dump(exclude_none=True),
            "_sort": "-date",
            "_count": 200,
            "_format": "json",
        }
        res = requests.get(route, params=params)
        return res.json()

    return fhir_observation_search


def create_fhir_vitals_search(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_vitals_search(
        *,
        category: Annotated[str, 'Use "vital-signs" to search for vitals observations.'],
        patient: Annotated[str, "Reference to a patient resource the condition is for."],
        date: Annotated[str | None, "The date range for when the observation was taken."] = None,
    ) -> dict:
        """Observation.Search (Vitals) retrieves vital sign / flowsheet observations."""
        route = f"{fhir_api_base}Observation"
        args = VitalsSearchParams(category=category, patient=patient, date=date)
        params = {
            **args.model_dump(exclude_none=True),
            "_format": "json",
        }
        res = requests.get(route, params=params)
        return res.json()

    return fhir_vitals_search


def create_fhir_procedure_search(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_procedure_search(
        *,
        search_params: Annotated[ProcedureSearchParams, "FHIR Procedure search parameters (completed procedures)."],
        explanation: Annotated[str, "Explanation for calling this tool."],
    ) -> dict:
        """Procedure.Search (Orders) returns completed Procedure resources."""
        route = f"{fhir_api_base}Procedure"
        search_params_obj = ProcedureSearchParams.model_validate(search_params)
        params = {
            **search_params_obj.model_dump(exclude_none=True),
            "_sort": "-date",
            "_count": 200,
        }
        res = requests.get(route, params=params)
        return res.json()

    return fhir_procedure_search


def create_fhir_condition_search(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_condition_search(
        *,
        search_params: Annotated[ConditionSearchParams, "FHIR Condition search parameters (problem list)."],
        explanation: Annotated[str, "Explanation for calling this tool."],
    ) -> dict:
        """Condition.Search (Problems) retrieves reconciled/confirmed problem list items."""
        route = f"{fhir_api_base}Condition"
        search_params_obj = ConditionSearchParams.model_validate(search_params)
        params = {
            **search_params_obj.model_dump(exclude_none=True),
            "_count": 200,
            "_format": "json",
        }
        res = requests.get(route, params=params)
        return res.json()

    return fhir_condition_search


def create_fhir_medication_request_search(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_medication_request_search(
        *,
        patient: Annotated[str, "The FHIR patient ID."],
        category: Annotated[
            str | None,
            "Medication category filter. Supported: Inpatient, Outpatient, Community, Discharge.",
        ] = None,
        date: Annotated[
            str | None,
            "Medication administration date filter (FHIR date parameter format).",
        ] = None,
    ) -> dict:
        """MedicationRequest.Search queries for medication orders for a patient."""
        route = f"{fhir_api_base}MedicationRequest"
        args = MedicationRequestSearchParams(patient=patient, category=category, date=date)
        params = {
            **args.model_dump(exclude_none=True),
            "_count": 200,
            "_format": "json",
        }
        res = requests.get(route, params=params)
        return res.json()

    return fhir_medication_request_search


def create_fhir_vitals_create(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_vitals_create(
        *,
        resourceType: Annotated[str, 'Use "Observation" for vitals observations.'],
        category: Annotated[list[Category], "Observation category coding. Use vital-signs."],
        code: Annotated[CodeText, "The flowsheet ID / mapping / code of what is being measured."],
        effectiveDateTime: Annotated[str, "The date/time the observation was taken, in ISO format."],
        status: Annotated[str, 'The status of the observation. Only a value of "final" is supported.'],
        valueString: Annotated[str, "Measurement value."],
        subject: Annotated[SubjectReference, "Patient reference: Patient/{patient_id}."],
    ) -> dict:
        """Observation.Create (Vitals) files vital signs / flowsheet values."""
        route = f"{fhir_api_base}Observation"
        args = VitalsCreateParams(
            resourceType=resourceType,
            category=category,
            code=code,
            effectiveDateTime=effectiveDateTime,
            status=status,
            valueString=valueString,
            subject=subject,
        )
        res = requests.post(
            route,
            json=args.model_dump(exclude_none=True),
            headers={"Content-Type": "application/fhir+json"},
        )
        res.raise_for_status()
        return res.json()

    return fhir_vitals_create


def create_fhir_medication_request_create(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_medication_request_create(
        *,
        resourceType: Annotated[str, 'Use "MedicationRequest" for medication requests.'],
        medicationCodeableConcept: Annotated[
            MedicationCodeableConcept, "Medication to order (coding system/code/display + free-text)."
        ],
        authoredOn: Annotated[str, "The date the prescription was written, in ISO format."],
        dosageInstruction: Annotated[list[DosageInstruction], "Dosing instructions."],
        status: Annotated[str, 'Medication request status. Use "active".'],
        intent: Annotated[str, 'Medication request intent. Use "order".'],
        subject: Annotated[SubjectReference, "Patient reference: Patient/{patient_id}."],
    ) -> dict:
        """MedicationRequest.Create files a medication order for a patient."""
        route = f"{fhir_api_base}MedicationRequest"
        args = MedicationRequestCreateParams(
            resourceType=resourceType,
            medicationCodeableConcept=medicationCodeableConcept,
            authoredOn=authoredOn,
            dosageInstruction=dosageInstruction,
            status=status,
            intent=intent,
            subject=subject,
        )
        res = requests.post(
            route,
            json=args.model_dump(exclude_none=True),
            headers={"Content-Type": "application/fhir+json"},
        )
        res.raise_for_status()
        return res.json()

    return fhir_medication_request_create


def create_fhir_service_request_create(fhir_api_base: str) -> Callable[..., dict]:
    def fhir_service_request_create(
        *,
        resourceType: Annotated[str, 'Use "ServiceRequest" for service requests.'],
        code: Annotated[Code, "Service/procedure code (e.g., CPT) to order."],
        authoredOn: Annotated[str, "The order instant (signed/signed+held), in ISO format."],
        status: Annotated[str, 'ServiceRequest status. Use "active".'],
        intent: Annotated[str, 'ServiceRequest intent. Use "order".'],
        priority: Annotated[str, 'ServiceRequest priority. Use "stat".'],
        subject: Annotated[SubjectReference, "Patient reference: Patient/{patient_id}."],
        note: Annotated[Note, "Free-text comment / indication for the order."],
        occurrenceDateTime: Annotated[str, "When the service should occur, in ISO format."],
    ) -> dict:
        """ServiceRequest.Create (Order) files an order for a patient."""
        route = f"{fhir_api_base}ServiceRequest"
        args = ServiceRequestCreateParams(
            resourceType=resourceType,
            code=code,
            authoredOn=authoredOn,
            status=status,
            intent=intent,
            priority=priority,
            subject=subject,
            note=note,
            occurrenceDateTime=occurrenceDateTime,
        )
        res = requests.post(
            route,
            json=args.model_dump(exclude_none=True),
            headers={"Content-Type": "application/fhir+json"},
        )
        res.raise_for_status()
        return res.json()

    return fhir_service_request_create


def create_finish() -> Callable[..., list[Any]]:
    def finish(*, value: list[Any]) -> list[Any]:
        """Finish the episode by returning the final answer as a JSON-serializable list."""
        return value

    return finish


def calculator(*, expression: str) -> str:
    """Evaluate a mathematical expression using a restricted evaluator (no I/O or imports)."""
    expression = (expression or "").strip()
    if not expression:
        return "Error: Empty expression"
    if len(expression) > 500:
        return "Error: Expression too long"
    if not re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\,\s\*\^a-zA-Z_:\'\"=<>]+", expression):
        return "Error: Invalid characters"

    evaluator = SimpleEval(
        names={"math": math, "datetime": datetime_module},
        functions={"Decimal": Decimal, "sum": sum},
    )
    evaluator.ATTR_INDEX_FALLBACK = None
    try:
        result = evaluator.eval(expression)
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


def build_tool_list(fhir_api_base: str) -> list[Callable[..., Any]]:
    return [
        create_patient_search(fhir_api_base),
        create_fhir_observation_search(fhir_api_base),
        create_fhir_vitals_search(fhir_api_base),
        create_fhir_procedure_search(fhir_api_base),
        create_fhir_condition_search(fhir_api_base),
        create_fhir_medication_request_search(fhir_api_base),
        create_fhir_vitals_create(fhir_api_base),
        create_fhir_medication_request_create(fhir_api_base),
        create_fhir_service_request_create(fhir_api_base),
        calculator,
        create_finish(),
    ]
