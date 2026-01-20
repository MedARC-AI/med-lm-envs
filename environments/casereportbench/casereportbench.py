"""CaseReportBench Environment

CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports
Official repository: https://github.com/cindyzhangxy/CaseReportBench
Hugging Face: https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark


@inproceedings{zhang2025casereportbench,
title={CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports},
author={Zhang, Xiao Yu Cindy and Ferreira, Carlos R. and Rossignol, Francis and Ng, Raymond T. and Wasserman, Wyeth and Zhu, Jian},
booktitle={Proceedings of the Sixth Conference on Health, Inference, and Learning},
year={2025},
publisher={PMLR}
}
"""

import json
import logging
from enum import Enum
from typing import Any, Iterable, List

import numpy as np
import verifiers as vf
from datasets import Dataset, load_dataset
from medarc_verifiers.parsers.json_parser import JSONParser

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

try:
    from Levenshtein import distance
except ImportError:
    distance = None

try:
    import evaluate
except ImportError:
    evaluate = None

logger = logging.getLogger(__name__)

# =============================================================================
# PREGNANCY 
# =============================================================================
PREGNANCY_DOCSTRING = """Extract detailed information from medical texts focusing on neonatal and maternal health outcomes and pregnancy complications. This class is designed to capture observations, health assessments, and conditions specific to pregnancy, explicitly excluding treatment details and any unrelated systemic conditions."""

PREGNANCY_DESC = """Organize pregnancy, neonatal, and maternal health information into a JSON-compliant dictionary."""

PREGNANCY_PREFIX = """Based on the provided text, extract and return information directly relevant to pregnancy, neonatal, and maternal health in a structured dictionary format. Include the following top-level keys:
                  - 'neonatal_health': List conditions or observations directly related to the infant's health. Use an empty list [] if no pertinent information is available.
                  - 'maternal_health': List conditions or observations directly related to the mother's health. Use an empty list [] if no pertinent information is available.
                  - 'prengancy_test_imaging_exam': List all maternal and neonate lab tests, genetics tests, physical exam, and diangostic 
                  imaging and their respective positive and negative results. Use an empty list [] if no complications are found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the pregnancy, maternal or neonatal observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "neonatal_health": ["Premature birth observed"],
                    "maternal_health": ["Gestational diabetes diagnosed"],
                    "prengancy_tests_image_exam": ["24 week ultrasound reveals normal fetal development"]
                  }
                  Example output if no relevant data is available:
                  {
                    "neonatal_health": [],
                    "maternal_health": [],
                    "pregnancy_tests_image_exam": []
                  }"""

# =============================================================================
# VITALS_HEMA 
# =============================================================================
VITALS_HEMA_DOCSTRING = """Extract vital signs and hematological observations from provided text, focusing solely on the measurement values and their units, and excluding any unrelated systemic conditions, treatments, or non-diagnostic content such as figures and tables."""

VITALS_HEMA_DESC = """Organize vital sign and hematological information into a JSON-compliant dictionary, focusing on precise and clinically relevant data."""

VITALS_HEMA_PREFIX = """Based on the provided text, extract and return vital signs and hematological information in a structured dictionary format. Include the following top-level keys:
                  - 'temperature': List the temperature values combined with units if mentioned. Use an empty list [] if no temperature data is available.
                  - 'pulse': List pulse rate values combined with units if mentioned. Use an empty list [] if no pulse data is available.
                  - 'respiratory_rate': List respiratory rate values combined with units if mentioned. Use an empty list [] if no respiratory data is available.
                  - 'blood_pressure': List blood pressure readings, specifying both systolic and diastolic values combined with units if mentioned. Use an empty list [] if no blood pressure data is available.
                  - 'oxygen_saturation (SpO2)': List oxygen saturation values combined with units if mentioned. Use an empty list [] if no SpO2 data is available.
                  - 'hematological_conditions': List conditions or observations directly related to the the blood system, which includes components like red blood cells, white blood cells, platelets, blood vessels, bone marrow, lymph nodes, and the proteins involved in bleeding and clotting.
                  - 'hematology_tests_measurements': List all hematological related measurements such as hemoglobin, hematocrit, white blood cell count, platelet count, and any other relevant blood test results. Use an empty list [] if no hematology data is available.
                  Ensure all entries are verbatim quotations from or clear deductions from the text, without making assumptions. 
                  Focus on diagnostic clarity and ensure that the output is directly relevant to the reasons for the medical visit. 
                  Exclude reference to any figures or tables.  
                  Example output for provided text containing relevant data:
                  {
                    "temperature": ["37.5°C"],
                    "pulse": ["72 bpm"],
                    "respiratory Rate": ["16 breaths per minute"],
                    "blood_pressure": ["120/80 mm Hg"],
                    "oxygen_saturation (SpO2)": ["98%"],
                    "hematological_conditions": ["Diagnosed with anemia"]
                    "hematology_tests_measurements": ["Hemoglobin: 13.5 g/dL", 
                    "WBC count: 6,000 /µL",   "Platelet count: 250,000 /µL"]
                  }
                  Example output if no relevant data is available:
                  {
                    "temperature": [],
                    "pulse": [],
                    "respiratory_rate": [],
                    "blood_pressure": [],
                    "oxygen_saturation (SpO2)": [],
                    "hematological_conditions": []
                    "hematology_tests_measurements": []
                  }"""

# =============================================================================
# NEURO 
# =============================================================================
NEURO_DOCSTRING = """Extract detailed information from medical texts focusing on assessments related to neurological and cognitive functions, as well as conditions specifically affecting the head. This class is designed to capture observations, diagnostic results, and imaging findings specific to these areas, explicitly excluding treatment details and any unrelated systemic conditions."""

NEURO_DESC = """Organize neurological, cognitive, and head information into a JSON-compliant dictionary, focusing exclusively on relevant diagnostic data."""

NEURO_PREFIX = """Based on the provided text, extract and return information directly relevant to neurological and cognitive functions as well as head-related conditions in a structured dictionary format. Include the following top-level keys:
                  - 'neurological': List observations and conditions directly related to the neurological system. Use an empty list [] if no pertinent information is available.
                  - 'cognitive': List observations and conditions directly related to cognitive functions. Use an empty list [] if no pertinent information is available.
                  - 'Neuro_Tests_Imaging_Exam': List descriptions of tests, measurements, physical exam, and diagnostic imaging specific to the neurological and cognitive areas, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  Ensure all entries are verbatim quotes from or clear deductions based from the provided text, without making assumptions or inference.  
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures and tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the neurological observations.
                  
                  Example output for provided text containing relevant data:
                  {
                    "neurological": ["Increased intracranial pressure observed"],
                    "cognitive": ["Impaired short-term memory noted"],
                    "Neuro_Tests_Imaging": ["MRI Brain: Evidence of cerebral atrophy"]
                  }
                  Example output if no relevant data is available:
                  {
                    "neurological": [],
                    "cognitive": [],
                    "Neuro_tests_image_exam": []
                  }"""

# =============================================================================
# EENT 
# =============================================================================
EENT_DOCSTRING = """Extract specific EENT (Eyes, Ears, Nose, and Throat) information from medical text, focusing on conditions and symptoms directly affecting these areas without making inferences about related systemic conditions."""

EENT_DESC = """Extract and organize EENT information into a JSON-compliant dictionary, focusing on observations and symptoms directly relevant to EENT structures. Exclude all treatment details and conditions not directly involving the EENT structures."""

EENT_PREFIX = """Based on the provided text, extract and return information directly relevant to EENT conditions in a structured dictionary format. Include observations or symptoms for 'eyes', 'ears', 'nose', and 'throat', using the following top-level keys:
                  - 'eyes': Include only direct observations or symptoms related to eye conditions. Use an empty list [] if no information is available.
                  - 'ears': Include only direct observations or symptoms related to ear conditions. Use an empty list [] if no information is available.
                  - 'nose': Include only direct observations or symptoms related to nose conditions. Use an empty list [] if no information is available.
                  - 'throat': Include only direct observations or symptoms related specifically to throat conditions. Use an empty list [] if no information is available.
                  - 'EENT_tests_image_exam': Include a sub-dictionary for EENT-specific tests, measurements, physical exam, and diagnostic imaging, detailing their respective positive and negative findings if available. Use an empty list [] if no relevant tests or imaging results are mentioned.
                  Ensure all entries are verbatim or clear deductions from the provided text without making assumptions. 
                  all entries should focus on diagnostic clarity and exclude any treatment details, references to figures or tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the ears, eyes, nose and throat observations.
                  
                  Here is an example of the output format:
{
  "eyes": ["Detailed description of eye symptoms from the text"],
  "ears": ["Detailed description of ear symptoms from the text"],
  "nose": ["Detailed description of nasal symptoms from the text"],
  "throat": ["Detailed description of throat symptoms from the text"],
  "EENT_tests_image_exam": {"Test Name": "Specific findings from the test relevant to EENT"}
}

 Example output if no relevant data is available:
                  {
                    "eyes": [],
                    "ears": [],
                    "nose": [],
                    "throat": [],
                    "vascular": [],
                    "EENT_tests_image_exam": []
"""

# =============================================================================
# CVS 
# =============================================================================
CVS_DOCSTRING = """Extract specific information about the cardiovascular and vascular systems from medical texts, focusing on observable or reported data related to heart and blood vessel functions without including treatment details or unrelated systemic conditions."""

CVS_DESC = """Extract and organize cardiovascular and vascular system information into a JSON-compliant dictionary, focusing solely on relevant details. Exclude any treatment data or unrelated systemic condition references."""

CVS_PREFIX = """Based on the provided text, extract and return relevant information in a structured dictionary format. Include the following top-level keys:
                  - 'cardiac': List observations, signs, symptoms, or conditions directly related to the heart and its immediate functions also include cardiocerebral conditions such as stroke. Use an empty list [] if no pertinent information is available.
                  - 'vascular': List observations, signs, symptoms, or conditions directly related to the blood vessels and circulatory system. Use an empty list [] if no applicable data is found.
                  - 'CVS_tests_image_exam': List all cardiovascular lab tests, genetics tests, physical exam, and diangostic imaging and their respective positive and negative results if available. Use an empty list [] if no complications are found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the cardiovascular observations.
        
                  {
                    "cardiac": ["Elevated heart rate observed during stress test"],
                    "vascular": ["Visible varicose veins", "Signs of peripheral arterial disease"],
                    "CVS_tests_image_exam": ["Echocardiogram shows mild hypertrophy", "Carotid ultrasound revealed Plaque buildup noted", "Endocardiogram was performed"]
                  }
                  Example output if no relevant data is available:
                  {
                    "cardiac": [],
                    "vascular": [],
                    "CVS_tests_image_exam": []
                  }"""

# =============================================================================
# RESP 
# =============================================================================
RESP_DOCSTRING = """Extract specific information about the respiratory system from medical texts, focusing on observable or reported data related to respiratory functions without including treatment details or unrelated systemic conditions."""

RESP_DESC = """Extract and organize respiratory system information into a JSON-compliant dictionary, focusing solely on relevant details. Exclude any treatment data or unrelated systemic condition references."""

RESP_PREFIX = """Based on the provided text, extract and return relevant respiratory system information in a structured dictionary format. Include the following top-level keys:
                  - 'respiratory': List observations, signs, symptoms, or conditions directly related to the respiratory system. Use an empty list [] if no pertinent information is available.
                  - 'RESP_tests_image_exam': List descriptions of tests, measurements, physical exams, and diagnostic imaging specific to the respiratory system, including both positive and negative findings. Use an empty list [] if no applicable data is found. 
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the respiratory observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "respiratory": ["Increased respiratory rate observed during examination"],
                    "RESP_tests_image_exam": ["Spirometry: Reduced lung capacity", "Chest X-ray: No visible abnormalities"]
                  }
                  Example output if no relevant data is available:
                  {
                    "respiratory": [],
                    "RESP_tests_image_exam": []
                  }"""

# =============================================================================
# GI 
# =============================================================================
GI_DOCSTRING = """Extract specific information about the gastrointestinal system from medical texts, focusing on observable or reported data related to gastrointestinal functions without including treatment details or unrelated systemic conditions."""

GI_DESC = """Extract and organize gastrointestinal system information into a JSON-compliant dictionary, focusing solely on relevant details. Exclude any treatment data or unrelated systemic condition references."""

GI_PREFIX = """Based on the provided text, extract and return relevant gastrointestinal system information in a structured dictionary format. Include the following top-level keys:
                  - 'gastrointestinal': List observations, signs, symptoms, or conditions directly related to the gastrointestinal system. Use an empty list [] if no pertinent information is available.
                  - 'GI_tests_image_exam': List descriptions of tests, measurements, physical exams and diagnostic imaging specific to the gastrointestinal system, including both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  Ensure all entries are verbatim or clear deductions from the provided text without making assumptions. 
                  All etnries should focus on diagnostic clarity. Exclude any treatment details and exclude references to figures or tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the gastrointestinal observations.
               
                  Example output for provided text containing relevant data:
                  {
                    "gastrointestinal": ["Increased abdominal discomfort noted during examination"],
                    "GI_tests_image_exam": ["Colonoscopy: Evidence of polyps", "Abdominal Ultrasound: Normal liver and gallbladder morphology"]
                  }
                  Example output if no relevant data is available:
                  {
                    "gastrointestinal": [],
                    "GI_tests_image_exam": []
                  }"""

# =============================================================================
# GU 
# =============================================================================
GU_DOCSTRING = """Extract detailed information from medical texts focusing on assessments related to the genital and urinary systems. This class is designed to capture observations, diagnostic results, and imaging findings specific to these systems, explicitly excluding treatment details and any unrelated systemic conditions."""

GU_DESC = """Organize genitourinary system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions."""

GU_PREFIX = """Based on the provided text, extract and return the relevant genitourinary system information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'urinary': List observations and conditions directly related to the urinary tract. Use an empty list [] if no pertinent information is available.
                  - 'genital': List observations and conditions directly related to the genital organs. Use an empty list [] if no pertinent information is available.
                  - 'GU_tests_image_exam':  List descriptions of tests, measurements, physical exams and diagnostic imaging specific to the genital and urinary systems, detailing both positive and negative findings if available. 
                  Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the genitourinary observations.          
                  Example output for provided text containing relevant data:
                  {
                    "urinary": ["Bladder was full"],
                    "genital": ["Prostate enlargement noted"],
                    "GU_tests_image_exam": ["Ultrasound Kidney: No stones detected", "Bladder Ultrasound: Normal bladder wall thickness"]
                  }
                  Example output if no relevant data is available:
                  {
                    "urinary": [],
                    "genital": [],
                    "GU_tests_image_exam": []
                  }"""

# =============================================================================
# DERM  
# =============================================================================
DERM_DOCSTRING = """Extract detailed information from medical texts focusing on assessments related to the skin, facial features, and breast conditions. This class is designed to capture observations, diagnostic results, and imaging findings specific to dermatological assessments, explicitly excluding treatment details and any unrelated systemic conditions."""

DERM_DESC = """Organize dermatological information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions."""

DERM_PREFIX = """Based on the provided text, extract and return the relevant dermatological information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'skin_conditions': List observations and conditions directly related to the skin. Use an empty list [] if no pertinent information is available.
                  - 'facial_features': List observations and conditions directly related to the facial features. Use an empty list [] if no pertinent information is available.
                  - 'breast_conditions': List observations and conditions directly related to the breasts. Use an empty list [] if no pertinent information is available.
                  - 'derm_breasts_facial_tests_image_exam': List descriptions of tests, measurements, physical exams and diagnostic imaging specific to dermatological, breasts and facial feature assessments, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the dermatology or facial or breasts observations.
        
                  
                  Example output for provided text containing relevant data:
                  {
                    "skin_conditions": ["Psoriasis noted", "Severe acne observed"],
                    "facial_features": ["Rosacea on cheeks"],
                    "breast_conditions": ["breast looks normal"],
                    "derm_breasts_facial_tests_image_exam": ["Dermatoscopy: Melanocytic nevus identified", "Skin biopsy: Basal cell carcinoma confirmed", "mammography": "unremarkable fidings"]
                  }
                  Example output if no relevant data is available:
                  {
                    "skin_conditions": [],
                    "facial_features": [],
                    "breast_conditions": [],
                    "derm_breasts_facial_tests_image_exam": []
                  }"""

# =============================================================================
# MSK           
# =============================================================================
MSK_DOCSTRING = """Extract detailed information from medical texts focusing on assessments related to the musculoskeletal system, specifically targeting muscle and skeletal structures separately. This class is designed to capture observations, diagnostic results, and imaging findings specific to muscles and bones, explicitly excluding treatment details and any unrelated systemic conditions."""

MSK_DESC = """Organize musculoskeletal system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data for muscles and skeletal structures. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions."""

MSK_PREFIX = """Based on the provided text, extract and return the relevant musculoskeletal system information in a structured dictionary format. Organize the information into the following top-level keys:
                  - 'muscle': List observations and conditions directly related to muscle health and function. Use an empty list [] if no pertinent muscle information is available.
                  - 'skeletal': List observations and conditions directly related to the skeletal system, including bones and joints. Use an empty list [] if no pertinent skeletal information is available.
                  - 'MSK_tests_image_exam': List descriptions of tests, measurements, and diagnostic imaging that are specific to the musculoskeletal system, covering both muscles and bones, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the musculoskeletal observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "muscle": ["Muscle stiffness and pain reported"],
                    "skeletal": ["Bone density reduction noted", "Joint swelling observed"],
                    "MSK_tests_image_exam": ["MRI: Ligament tear detected", "Bone scan: Signs of osteoporosis"]
                  }
                  Example output if no relevant data is available:
                  {
                    "muscle": [],
                    "skeletal": [],
                    "MSK_tests_image_exam": []
                  }"""

# =============================================================================
# LYMPH
# =============================================================================
LYMPH_DOCSTRING = """Extract detailed information from medical texts focusing on assessments related to the lymphatic system. This class is designed to capture observations, diagnostic results, and imaging findings specific to components of the lymphatic system, explicitly excluding treatment details and any unrelated systemic conditions."""

LYMPH_DESC = """Organize lymphatic system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions."""

LYMPH_PREFIX = """Based on the provided text, extract and return the relevant lymphatic system information in a structured dictionary format. 
        Exclude reference to figures, tables or citations. Separate the information into the following top-level keys:
                  - 'adenoid': List observations and conditions directly related to adenoids. Use an empty list [] if no pertinent information is available.
                  - 'tonsils': List observations and conditions directly related to the tonsils. Use an empty list [] if no pertinent information is available.
                  - 'lymphatic_tissues': List observations and conditions directly related to general lymphatic tissues. Use an empty list [] if no pertinent information is available.
                  - 'lymph_nodes': List observations and conditions directly related to lymph nodes. Use an empty list [] if no pertinent information is available.
                  - 'thymus': List observations and conditions directly related to the thymus. Use an empty list [] if no pertinent information is available.
                  - 'bone_marrow': List observations and conditions directly related to bone marrow. Use an empty list [] if no pertinent information is available.
                  - 'spleen': List observations and conditions directly related to the spleen. Use an empty list [] if no pertinent information is available.
                  - 'immune_cells': List observations and conditions directly related to immune cells. Use an empty list [] if no pertinent information is available.
                  - 'Lymphatic_tests_image_exam': List descriptions of tests, measurements, physical exams and diagnostic imaging for any part of the lymphatic system, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the lymphatic system observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "adenoid": ["Enlarged adenoids observed"],
                    "tonsils": ["Tonsillitis diagnosed"],
                    "lymphatic_tissues": ["Signs of lymphedema noted"],
                    "lymph_nodes": ["Lymphadenopathy in cervical nodes"],
                    "thymus": ["Thymus hyperplasia found"],
                    "bone_marrow": ["Bone marrow biopsy shows increased cellularity"],
                    "spleen": ["Splenomegaly detected"],
                    "immune_cells": ["Increased leukocytes in blood test"],
                    "Lymphatic_tests_image_exam": ["PET scan: Abnormal lymph node activity"]
                  }
                  Example output if no relevant data is available:
                  {
                    "adenoid": [],
                    "tonsils": [],
                    "lymphatic_tissues": [],
                    "lymph_nodes": [],
                    "thymus": [],
                    "bone_marrow": [],
                    "spleen": [],
                    "immune_cells": [],
                    "Lymphatic_tests_image_exam": []
                  }"""

# =============================================================================
# ENDO 
# =============================================================================
ENDO_DOCSTRING = """Extract detailed information from medical texts focusing on assessments related to the endocrine system. This class is designed to capture observations, diagnostic results, and imaging findings specific to endocrine glands, explicitly excluding treatment details and any unrelated systemic conditions."""

ENDO_DESC = """Organize endocrine system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions."""

ENDO_PREFIX = """Based on the provided text, extract and return the relevant endocrine system information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'endocrine_glands': List observations and conditions directly related to specific endocrine glands such as the thyroid, pancreas, adrenal glands, pituitary gland, and others. Use an empty list [] if no pertinent information is available.
                  - 'Endocrine_tests_image_exam': List descriptions of tests, measurements, physical exams, and diagnostic imaging specific to the endocrine system, including both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the endocrine system observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "endocrine_glands": ["Thyroid enlargement noted", "Adrenal insufficiency observed"],
                    "Endocrine_tests_image_exam": ["Thyroid function test results: Elevated TSH", "CT scan: Adrenal mass detected"]
                  }
                  Example output if no relevant data is available:
                  {
                    "endocrine_glands": [],
                    "Endocrine_tests_image_exam": []
                  }"""

# =============================================================================
# HISTORY       
# =============================================================================
HISTORY_DOCSTRING = """Extract detailed historical information and cheif complaints from medical texts focusing on the patient's past medical and surgical history, history of present illness, social history, and family and genetics history, and chief complaint that brings patient to medical attention. This class is designed to capture comprehensive background data essential for diagnosis and treatment planning, explicitly excluding unrelated systemic conditions and treatment details."""

HISTORY_DESC = """Organize historical information into a JSON-compliant dictionary, focusing solely on relevant historical data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions."""

HISTORY_PREFIX = """Based on the provided text, extract and return the relevant historical information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'past_medical_history': List conditions and previous diagnoses related to the patient's past medical history. Use an empty list [] if no pertinent information is available.
                  - 'past_surgical_history': List past surgical interventions and outcomes. Use an empty list [] if no relevant surgical history is available.
                  - 'history_of_present_illness': List the chief complaint and detail the chronological development of the patient's current complaints and symptoms. Use an empty list [] if no detailed current illness history is provided.
                  - 'social_history': Include relevant lifestyle factors such as smoking, alcohol use, occupation, and living conditions. Use an empty list [] if no social history is available.
                  - 'family_and_genetics_history': List any known genetic conditions or diseases prevalent in the patient's family that might affect the patient's health. Use an empty list [] if no family or genetic history is available.
                   - 'chief complaint': List any known chief complaint that brought patient to seek medical attention. Use an empty list [] if no information. 
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the historical information provided.
        
                  Example output for provided text containing relevant data:
                  {
                    "past_medical_history": ["Diagnosed with hypertension", "Previous myocardial infarction"],
                    "past_surgical_history": ["Appendectomy in 2010", "Knee replacement in 2018"],
                    "history_of_present_illness": ["Gradual onset of chest pain over the past two months"],
                    "social_history": ["Smoker for 20 years, 10 cigarettes a day", "Works in construction"],
                    "family_and_genetics_history": ["Father had colon cancer", "Sister diagnosed with breast cancer at age 50"],
                    "chief_complaint":["Patient was brought to ER after the first episode of acute chest pain and hemoptysis"]
                  }
                  Example output if no relevant data is available:
                  {
                    "past_medical_history": [],
                    "past_surgical_history": [],
                    "history_of_present_illness": [],
                    "social_history": [],
                    "family_and_genetics_history": [],
                    "chief_complaint": [] 
                  }"""


# =============================================================================
# FULL PROMPT CONSTRUCTION - Combines all three components like DSPy does
# =============================================================================

def _build_full_prompt(docstring: str, desc: str, prefix: str) -> str:
    """Construct the full prompt as DSPy would format it.
    
    DSPy Signature prompts are constructed from:
    1. Class docstring (task context)
    2. OutputField desc (output format description)
    3. OutputField prefix (detailed instructions + few-shot examples)
    """
    return f"""{docstring}

{desc}

{prefix}"""


# Pre-built full prompts for each category
PREGNANCY_PROMPT = _build_full_prompt(PREGNANCY_DOCSTRING, PREGNANCY_DESC, PREGNANCY_PREFIX)
VITALS_HEMA_PROMPT = _build_full_prompt(VITALS_HEMA_DOCSTRING, VITALS_HEMA_DESC, VITALS_HEMA_PREFIX)
NEURO_PROMPT = _build_full_prompt(NEURO_DOCSTRING, NEURO_DESC, NEURO_PREFIX)
EENT_PROMPT = _build_full_prompt(EENT_DOCSTRING, EENT_DESC, EENT_PREFIX)
CVS_PROMPT = _build_full_prompt(CVS_DOCSTRING, CVS_DESC, CVS_PREFIX)
RESP_PROMPT = _build_full_prompt(RESP_DOCSTRING, RESP_DESC, RESP_PREFIX)
GI_PROMPT = _build_full_prompt(GI_DOCSTRING, GI_DESC, GI_PREFIX)
GU_PROMPT = _build_full_prompt(GU_DOCSTRING, GU_DESC, GU_PREFIX)
DERM_PROMPT = _build_full_prompt(DERM_DOCSTRING, DERM_DESC, DERM_PREFIX)
MSK_PROMPT = _build_full_prompt(MSK_DOCSTRING, MSK_DESC, MSK_PREFIX)
LYMPH_PROMPT = _build_full_prompt(LYMPH_DOCSTRING, LYMPH_DESC, LYMPH_PREFIX)
ENDO_PROMPT = _build_full_prompt(ENDO_DOCSTRING, ENDO_DESC, ENDO_PREFIX)
HISTORY_PROMPT = _build_full_prompt(HISTORY_DOCSTRING, HISTORY_DESC, HISTORY_PREFIX)


# =============================================================================
# PROMPT MAPPING - Maps task enum values to full prompts
# =============================================================================
PROMPTS = {
    "Vitals_Hema": VITALS_HEMA_PROMPT,
    "Neuro": NEURO_PROMPT,
    "EENT": EENT_PROMPT,
    "CVS": CVS_PROMPT,
    "RESP": RESP_PROMPT,
    "GI": GI_PROMPT,
    "GU": GU_PROMPT,
    "MSK": MSK_PROMPT,
    "DERM": DERM_PROMPT,
    "LYMPH": LYMPH_PROMPT,
    "History": HISTORY_PROMPT,
    "ENDO": ENDO_PROMPT,
    "Pregnancy": PREGNANCY_PROMPT,
}


def get_prompt(task: str) -> str:
    """Get the full prompt for a given task.
    
    Args:
        task: One of the 13 CaseReportBench task names.
        
    Returns:
        The complete prompt text including docstring, description, and few-shot examples.
        This matches the format DSPy would construct from the Signature class.
    """
    if task not in PROMPTS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {list(PROMPTS.keys())}")
    return PROMPTS[task]

def compute_normalized_token_set_ratio(list1: List[str], list2: List[str], method: str = "average") -> float:
    """
    Compute token set ratio normalized by the longer text length.
    Replicates compute_normalized_token_set_ratio from eval_metrics.py.
    """
    if not list1 and not list2:
        return 100.0
    if not list1 or not list2:
        return 0.0
    
    if fuzz is None:
        logger.warning("fuzzywuzzy not installed, TSR will be 0")
        return 0.0

    # Join items to calculate token length for normalization
    text1 = " ".join(list1)
    text2 = " ".join(list2)
    length1 = len(text1.split())
    length2 = len(text2.split())
    max_length = max(length1, length2, 1)

    # Compute raw token set ratios: for each item in list1, find max fuzzy match in list2
    scores = []
    for item1 in list1:
        item_scores = [fuzz.token_set_ratio(item1.lower(), item2.lower()) for item2 in list2]
        scores.append(max(item_scores) if item_scores else 0)

    # Aggregate raw scores
    if method == "average":
        raw_score = np.mean(scores)
    elif method == "max":
        raw_score = np.max(scores)
    else:
        raise ValueError("Invalid method. Choose 'average' or 'max'")

    # Normalize the raw score by the maximum text length
    normalized_score = (raw_score / max_length) * 100
    return min(normalized_score, 100.0)

def compute_normalized_levenshtein(list1: List[str], list2: List[str], method: str = "average") -> float:
    """
    Compute Levenshtein distance normalized by the longer text length.
    Replicates compute_normalized_levenshtein from eval_metrics.py.
    """
    if not list1 or not list2:
        return 0.0 if list1 != list2 else 1.0
    
    if distance is None:
        logger.warning("Levenshtein not installed, score will be 0")
        return 0.0

    scores = []
    for item1 in list1:
        # 1 - (dist / max_len)
        item_scores = [1 - (distance(item1, item2) / max(len(item1), len(item2), 1)) for item2 in list2]
        scores.append(max(item_scores) if item_scores else 0.0)

    if method == "average":
        return float(np.mean(scores))
    elif method == "max":
        return float(np.max(scores))
    else:
        raise ValueError("Invalid method. Choose 'average' or 'max'.")

def compute_exact_match(list1: List[str], list2: List[str]) -> float:
    """Compute exact match between two lists of strings."""
    return 1.0 if list1 == list2 else 0.0

def compute_bleu(pred_text: str, ref_text: str, k: int = 1) -> float:
    """Compute BLEU score using evaluate library."""
    if evaluate is None:
        return 0.0
    
    if not pred_text and not ref_text:
        return 1.0
    if not pred_text or not ref_text:
        return 0.0
    
    try:
        bleu_metric = evaluate.load("bleu")
        results = bleu_metric.compute(predictions=[pred_text], references=[[ref_text]])
        return float(results['precisions'][k-1])
    except Exception:
        return 0.0

def compute_rouge_l(pred_text: str, ref_text: str) -> float:
    """Compute ROUGE-L score using evaluate library."""
    if evaluate is None:
        return 0.0
    
    if not pred_text and not ref_text:
        return 1.0
    if not pred_text or not ref_text:
        return 0.0
    
    try:
        rouge_metric = evaluate.load("rouge")
        results = rouge_metric.compute(predictions=[pred_text], references=[[ref_text]])
        return float(results['rougeL'])
    except Exception:
        return 0.0




class CaseReportBenchTask(str, Enum):
    """The 14 clinical categories from the CaseReportBench dataset."""
    VITALS_HEMA = "Vitals_Hema"
    NEURO = "Neuro"
    EENT = "EENT"
    CVS = "CVS"
    RESP = "RESP"
    GI = "GI"
    GU = "GU"
    MSK = "MSK"
    DERM = "DERM"
    LYMPH = "LYMPH"
    HISTORY = "History"
    ENDO = "ENDO"
    PREGNANCY = "Pregnancy"
    ALL = "all"


def _flatten_to_strings(obj: Any) -> list[str]:
    """Flatten any nested JSON structure into a list of strings.
    
    This handles the various JSON structures in the ground truth annotations,
    including nested dicts and lists of varying depth.
    """
    if isinstance(obj, str):
        return [obj.strip()] if obj.strip() else []
    if isinstance(obj, list):
        res = []
        for item in obj:
            res.extend(_flatten_to_strings(item))
        return res
    if isinstance(obj, dict):
        res = []
        for val in obj.values():
            res.extend(_flatten_to_strings(val))
        return res
    return []


def _normalize_items(items: Iterable[str]) -> list[str]:
    """Remove duplicates and empty strings, preserving order."""
    seen = set()
    res = []
    for it in items:
        s = str(it).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            res.append(s)
    return res


def load_environment(
    task: str | CaseReportBenchTask = CaseReportBenchTask.ALL,
    max_examples: int = -1,
    **kwargs,
) -> vf.Environment:
    """Load the CaseReportBench environment for dense information extraction.
    
    This environment uses the VERBATIM prompts from the author's DSPy signatures
    (extractAug24.py) to ensure direct comparability to the paper's published results.
    
    Args:
        task: Which clinical category to evaluate. Use "all" for all 14 categories.
        max_examples: Maximum number of examples to load. -1 for all.
        **kwargs: Additional arguments passed to vf.SingleTurnEnv.
        
    Returns:
        A verifiers Environment configured for CaseReportBench evaluation.
    """
    
    # Load dataset from Hugging Face
    raw = load_dataset(
        "cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark", 
        split="train"
    )
    
    # Determine which tasks to load
    task_enum = CaseReportBenchTask(task) if isinstance(task, str) else task
    if task_enum == CaseReportBenchTask.ALL:
        tasks_to_load = [t for t in CaseReportBenchTask if t != CaseReportBenchTask.ALL]
    else:
        tasks_to_load = [task_enum]
    
    # Build examples
    examples = []
    for row in raw:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        pmcid = row.get("pmcid")
        
        for t in tasks_to_load:
            # Get ground truth annotations for this category
            gt_list = row.get(t.value)
            if not isinstance(gt_list, list):
                gt_list = []
            
            gt_items = _normalize_items(_flatten_to_strings(gt_list))
            
            # Get the VERBATIM prompt from author's DSPy signature
            author_prompt = get_prompt(t.value)
            
            # Construct the full prompt with case report text
            # This matches how DSPy would format the input
            prompt = (
                f"{author_prompt}\n\n"
                "---\n"
                "Medical text:\n"
                f"{text}\n"
                "---\n\n"
                "extract_info:"
            )
            
            examples.append({
                "question": prompt,
                "answer": json.dumps(gt_items, ensure_ascii=False),
                "info": {"text": text, "pmcid": pmcid, "task": t.value}
            })
    
    # Limit examples if requested
    if max_examples > 0:
        examples = examples[:max_examples]
    
    eval_dataset = Dataset.from_list(examples)
    
    # Parser for JSON output
    # Accepts multiple field names to handle variations in model output
    parser = JSONParser(
        fields=[("extractions", "findings", "output")], 
        answer_field="extractions"
    )
    
    def _parse_items(content: Any) -> list[str]:
        """Parse model output into a list of extracted items."""
        if not content:
            return []
        if isinstance(content, list):
            return _normalize_items(_flatten_to_strings(content))
        # Handle string completion - try to parse as JSON
        try:
            parsed = parser.parse(str(content), strip=True)
            if parsed:
                return _normalize_items(_flatten_to_strings(parsed))
        except Exception:
            pass
        return []

    # =========================================================================
    # REWARD FUNCTIONS - Strictly following paper's metrics
    # =========================================================================

    def token_set_ratio_reward(parser, completion, answer, **kwargs) -> float:
        """Primary reward: Token Set Ratio (normalized).
        
        This is the paper's signature metric for fuzzy clinical match.
        Replicates compute_normalized_token_set_ratio from eval_metrics.py.
        """
        pred_items = _parse_items(completion)
        gt_items = json.loads(answer)
        # TSR returns 0-100, normalize to 0-1 for reward
        return compute_normalized_token_set_ratio(pred_items, gt_items) / 100.0

    def bleu1_reward(parser, completion, answer, **kwargs) -> float:
        """BLEU-1: 1-gram precision."""
        pred_items = _parse_items(completion)
        gt_items = json.loads(answer)
        return compute_bleu(" ".join(pred_items), " ".join(gt_items), k=1)

    def bleu4_reward(parser, completion, answer, **kwargs) -> float:
        """BLEU-4: 4-gram precision."""
        pred_items = _parse_items(completion)
        gt_items = json.loads(answer)
        return compute_bleu(" ".join(pred_items), " ".join(gt_items), k=4)

    def rougeL_reward(parser, completion, answer, **kwargs) -> float:
        """ROUGE-L: Longest Common Subsequence overlap."""
        pred_items = _parse_items(completion)
        gt_items = json.loads(answer)
        return compute_rouge_l(" ".join(pred_items), " ".join(gt_items))

    def omission_reward(parser, completion, answer, **kwargs) -> float:
        """Paper's Omission metric: penalize if ref has info but LLM has none.
        
        Returns 0.0 if omission occurred, 1.0 otherwise.
        """
        pred_items = _parse_items(completion)
        gt_items = json.loads(answer)
        ref_has_info = len(gt_items) > 0
        llm_empty = len(pred_items) == 0
        is_omission = ref_has_info and llm_empty
        return 0.0 if is_omission else 1.0

    def hallucination_reward(parser, completion, answer, **kwargs) -> float:
        """Paper's Hallucination metric: penalize if ref empty but LLM has info.
        
        Returns 0.0 if hallucination occurred, 1.0 otherwise.
        """
        pred_items = _parse_items(completion)
        gt_items = json.loads(answer)
        ref_empty = len(gt_items) == 0
        llm_has_info = len(pred_items) > 0
        is_hallucination = ref_empty and llm_has_info
        return 0.0 if is_hallucination else 1.0

    # Create rubric with strict adherence to paper's metrics
    # TSR is the primary metric (weight=1.0), others are logged (weight=0.0)
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(token_set_ratio_reward, weight=1.0)  # Primary
    rubric.add_reward_func(bleu1_reward, weight=0.0)            # Logged
    rubric.add_reward_func(bleu4_reward, weight=0.0)            # Logged
    rubric.add_reward_func(rougeL_reward, weight=0.0)           # Logged
    rubric.add_reward_func(omission_reward, weight=0.0)         # Logged
    rubric.add_reward_func(hallucination_reward, weight=0.0)    # Logged

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt="You are a specialized medical information extraction assistant.",
        rubric=rubric,
        parser=parser,
        **kwargs
    )
