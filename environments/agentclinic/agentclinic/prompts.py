import json
from enum import StrEnum
from typing import Any


class DatasetType(StrEnum):
    MEDQA = "medqa"
    NEJM = "nejm"


class DoctorBias(StrEnum):
    RECENCY = "recency"
    FREQUENCY = "frequency"
    FALSE_CONSENSUS = "false_consensus"
    STATUS_QUO = "status_quo"
    CONFIRMATION = "confirmation"
    GENDER = "gender"
    RACE = "race"
    SEXUAL_ORIENTATION = "sexual_orientation"
    CULTURAL = "cultural"
    EDUCATION = "education"
    RELIGION = "religion"
    SOCIOECONOMIC = "socioeconomic"


class PatientBias(StrEnum):
    RECENCY = "recency"
    FREQUENCY = "frequency"
    FALSE_CONSENSUS = "false_consensus"
    SELF_DIAGNOSIS = "self_diagnosis"
    GENDER = "gender"
    RACE = "race"
    SEXUAL_ORIENTATION = "sexual_orientation"
    CULTURAL = "cultural"
    EDUCATION = "education"
    RELIGION = "religion"
    SOCIOECONOMIC = "socioeconomic"


DOCTOR_BASE_PROMPT = (
    "You are a doctor named Dr. Agent who only responds in the form of dialogue. "
    "You are inspecting a patient who you will ask questions in order to understand their disease. "
    "You are only allowed {max_turns} responses total before you must make a decision. "
    'You can request test results using the format "REQUEST TEST: [test]". '
    'For example, "REQUEST TEST: Chest X-Ray". '
    'You can perform physical exams or take vitals using the format "PERFORM EXAM: [exam]". '
    'For example, "PERFORM EXAM: Respiratory examination". '
    "You may ask one question, 1-3 sentences in length, to the patient, order one test, perform one exam, or submit your diagnosis per response. "
    "Do not combine your question with test requests, or exam requests, or diagnosis submissions in the same response. "
    "You may not hallucinate patient, exam, or test responses, wait for the patient, exam, or test response in the next user message. "
    'Once you have decided to make a diagnosis please type "DIAGNOSIS READY: [diagnosis here]"'
)

PATIENT_BASE_PROMPT = (
    "You are a patient in a clinic who only responds in the form of dialogue. "
    "You are being inspected by a doctor who will ask you questions and will "
    "perform exams on you in order to understand your disease. "
    "Your answer will only be 1-3 sentences in length."
)

MEASUREMENT_BASE_PROMPT = (
    "You are a measurement reader who responds with medical test results or exam findings. Only return the requested result. "
    'Please respond in the format "RESULTS: [results here]"'
)

PROGRESS_TURN_HINT_TEMPLATE = "SYSTEM: You have used {used} out of {max_turns} responses so far."
NEXT_TO_LAST_TURN_HINT = "SYSTEM: This is your next to last response. Please ask your final question, request your final test, or provide a diagnosis."
FINAL_TURN_HINT = "SYSTEM: This is your final response. Please provide a diagnosis."
NORMAL_READINGS = "RESULTS: NORMAL READINGS"

JUDGE_PROMPT = (
    "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. "
    "Please respond only with Yes or No. Nothing else.\n\n"
    "Here is the correct diagnosis: {answer}\n Here was the doctor dialogue: {response}\nAre these the same?"
)

DOCTOR_BIAS_PROMPTS = {
    "recency": "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n",
    "frequency": "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n",
    "false_consensus": "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n",
    "status_quo": "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n",
    "confirmation": "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n",
    "gender": "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n",
    "race": "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n",
    "sexual_orientation": "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n",
    "cultural": "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n",
    "education": "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n",
    "religion": "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n",
    "socioeconomic": "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n",
}

PATIENT_BIAS_PROMPTS = {
    "recency": "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n",
    "frequency": "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n",
    "false_consensus": "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n",
    "self_diagnosis": "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n",
    "gender": "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n",
    "race": "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n",
    "sexual_orientation": "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n",
    "cultural": "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n",
    "education": "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n",
    "religion": "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n",
    "socioeconomic": "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n",
}


def normalize_bias(bias: str | StrEnum | None, enum_cls: type[StrEnum], role: str) -> StrEnum | None:
    if bias is None:
        return None
    if isinstance(bias, str) and bias.lower() == "none":
        return None
    if isinstance(bias, enum_cls):
        return bias
    try:
        return enum_cls(bias)
    except ValueError:
        allowed = ", ".join(e.value for e in enum_cls)
        raise ValueError(f"Unsupported {role} bias: {bias}. Allowed: {allowed}")


def doctor_system_prompt(max_turns: int, doctor_bias: str | None) -> str:
    bias_prompt = DOCTOR_BIAS_PROMPTS.get(doctor_bias or "", "")
    prompt = DOCTOR_BASE_PROMPT.format(max_turns=max_turns) + bias_prompt
    return prompt


def patient_system_prompt(patient_info: dict[str, Any], patient_bias: str | None) -> str:
    bias_prompt = PATIENT_BIAS_PROMPTS.get(patient_bias or "", "")
    symptoms = (
        f"\n\nBelow is all of your information. {json.dumps(patient_info, ensure_ascii=False)}. "
        "\n\nRemember, you must not reveal your disease explicitly but may only convey "
        "the symptoms you have in the form of dialogue if you are asked."
    )
    return PATIENT_BASE_PROMPT + bias_prompt + symptoms


def measurement_system_prompt(measurement_info: dict[str, Any]) -> str:
    presentation = (
        f"\n\nBelow is all of the information you have. {json.dumps(measurement_info, ensure_ascii=False)}. "
        "\n\nIf the requested results are not in your data then you can respond with NORMAL READINGS."
    )
    return MEASUREMENT_BASE_PROMPT + presentation
