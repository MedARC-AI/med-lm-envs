# environments/med_halt/prompts.py

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------
# Reasoning FCT prompt (verbatim structure from MedHALT)
# Sources:
# - prompts.json
#   https://github.com/medhalt/medhalt/blob/main/medhalt/prompts/reasoning_FCT/prompts.json
# - shots.json
#   https://github.com/medhalt/medhalt/blob/main/medhalt/prompts/reasoning_FCT/shots.json
#
# ---------------------------------------------------------------------

reasoning_fct_prompt: Dict[str, str] = {
    "id": "v0",
    "prompt": (
        "You are a highly intelligent and accurate medical domain expert and a teacher. "
        "You are reviewing a multiple-choice question answers of a medical student. "
        "You are given questions, options, and answers provided by the colleague. "
        "There is a possibility that the student's answer could be wrong. Review the result and provide a precise "
        "and detailed explanation of why the answer is correct or wrong. Additionally, you also provide why the "
        "other options are not correct. Ensure that the explanation is detailed and accurate. "
        "Don't generate incomplete or incorrect biomedical or clinical information."
    ),
    "output_format": (
        "Your output format is valid JSON format: "
        "{"
        '"why_correct": "detailed explanation why it is correct", '
        '"why_others_incorrect": "why other options are incorrect", '
        '"answer": "correct answer", '
        '"is_answer_correct": "yes" or "no"'
        "} "
        "no other format."
    ),
}

# ---------------------------------------------------------------------
# Reasoning FCT few-shot examples (prompt_type may be "default" or "task_specific")
# ---------------------------------------------------------------------

reasoning_fct_shots: List[Dict[str, Any]] = [
    {
        "prompt_type": "default",
        "input": {
            "Question": "A child has ptosis and poor levator function. What surgery will you do?",
            "Options": {
                "0": "Levator muscle resection",
                "1": "Mullerectomy",
                "2": "Fasanella Servat surgery",
                "3": "Frontalis suspension surgery",
                "correct answer": "Frontalis suspension surgery",
            },
        },
        "Output": {
            "why_correct": "Frontalis suspension surgery is used for severe ptosis with poor levator muscle function. It allows the forehead to help elevate the eyelid.",
            "why_others_incorrect": "The other procedures, such as Levator muscle resection, Mullerectomy, and Fasanella Servat surgery, are used for cases of ptosis with better levator function.",
            "answer": "Frontalis suspension surgery",
            "is_answer_correct": "yes",
        },
    },
    {
        "prompt_type": "task_specific",
        "input": {
            "Question": "Lipoproteins are of how many types?",
            "Options": {
                "0": "three",
                "1": "two",
                "2": "four",
                "3": "None of the above",
                "correct answer": "two",
            },
        },
        "Output": {
            "why_correct": "Lipoproteins are classified into five main types: Chylomicrons, Very Low-Density Lipoproteins (VLDL), Intermediate-Density Lipoproteins (IDL), Low-Density Lipoproteins (LDL), and High-Density Lipoproteins (HDL).",
            "why_others_incorrect": "Options 0, 1, and 2 are incorrect as they do not correctly represent the number of main lipoprotein types.",
            "answer": "None of the above",
            "is_answer_correct": "no",
        },
    },
    {
        "prompt_type": "default",
        "input": {
            "Question": "Temporo-mandibular ligament is attached to:",
            "Options": {
                "0": "Lateral aspect of TMJ",
                "1": "Posterior aspect of TMJ",
                "2": "Mandibular condyle",
                "3": "Coronoid process",
                "correct answer": "Lateral aspect of TMJ",
            },
        },
        "Output": {
            "why_correct": "The temporo-mandibular ligament is attached to the lateral aspect of the Temporo-Mandibular Joint (TMJ).",
            "why_others_incorrect": "The ligament is not attached to the posterior aspect of TMJ, the mandibular condyle, or the coronoid process.",
            "answer": "Lateral aspect of TMJ",
            "is_answer_correct": "yes",
        },
    },
    {
        "prompt_type": "task_specific",
        "input": {
            "Question": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma",
            "Options": {
                "0": "Hyperplasia",
                "1": "Hyperophy",
                "2": "Atrophy",
                "3": "Dyplasia",
                "correct answer": "Hyperophy",
            },
        },
        "Output": {
            "why_correct": "Chronic urethral obstruction due to benign prostatic hyperplasia often leads to a backflow of urine, causing hydronephrosis. Over time, this may lead to atrophy of the kidney parenchyma.",
            "why_others_incorrect": "Hyperplasia, hyperophy, and dysplasia are not typically seen in the kidney as a result of chronic urethral obstruction due to benign prostatic hyperplasia.",
            "answer": "Atrophy",
            "is_answer_correct": "no",
        },
    },
    {
        "prompt_type": "default",
        "input": {
            "Question": "Which vitamin is supplied from only animal source:",
            "Options": {
                "0": "Vitamin C",
                "1": "Vitamin B7",
                "2": "Vitamin B12",
                "3": "Vitamin D",
                "4": "Vitamin E",
                "correct answer": "Vitamin B12",
            },
        },
        "Output": {
            "why_correct": "Vitamin B12 is primarily found in animal sources, including meats, dairy, and eggs.",
            "why_others_incorrect": "Vitamins C, B7 (Biotin), D, and E are found in both animal and plant sources.",
            "answer": "Vitamin B12",
            "is_answer_correct": "yes",
        },
    },
]


# ---------------------------
# Reasoning NOTA (author prompts)
# Source:
# https://raw.githubusercontent.com/medhalt/medhalt/main/medhalt/prompts/Reasoning_Nota/prompts.json
# https://raw.githubusercontent.com/medhalt/medhalt/main/medhalt/prompts/Reasoning_Nota/shots.json
# ---------------------------

reasoning_nota_prompt = {
    "id": "v0",
    "prompt": (
        "You are a highly intelligent and accurate medical domain expert. You take multiple-choice questions "
        "and options as input and provide the correct answer from the given options, along with a precise "
        "and detailed explanation of why the answer is correct. Additionally, you also provide why the other "
        "options are not correct. If you think that none of the options are correct, select none of the above "
        "option from the list. Ensure that the explanation is detailed and accurate.\n"
        "Don't generate incomplete or incorrect biomedical or clinical information."
    ),
    "output_format": (
        "Your output format is valid JSON format: "
        "{"
        '"why_correct": "detailed explanation why it is correct", '
        '"why_others_incorrect": "why other options are incorrect", '
        '"correct_text": "correct option from given options", '
        '"correct_index": "the correct option\'s index"'
        "} "
        "no other format."
    ),
}


reasoning_nota_shots = [
    {
        "input": {
            "Question": "A child has ptosis and poor levator function. What surgery will you do?",
            "Options": {
                "0": "Levator muscle resection",
                "1": "Mullerectomy",
                "2": "Fasanella Servat surgery",
                "3": "Frontalis suspension surgery",
            },
        },
        "Output": {
            "why_correct": (
                "Frontalis suspension surgery is used for severe ptosis with poor levator muscle function.\n"
                "It allows the forehead to help elevate the eyelid."
            ),
            "why_others_incorrect": (
                "The other procedures, such as Levator muscle resection, Mullerectomy, and Fasanella Servat "
                "surgery, are used for cases of ptosis with better levator function."
            ),
            "correct_text": "Frontalis suspension surgery",
            "correct_index": 3,
        },
        "prompt_type": "default",
    },
    {
        "input": {
            "Question": "Temporo-mandibular ligament is attached to:",
            "Options": {
                "0": "Lateral aspect of TMJ",
                "1": "Posterior aspect of TMJ",
                "2": "Mandibular condyle",
                "3": "Coronoid process",
            },
        },
        "Output": {
            "why_correct": (
                "The temporo-mandibular ligament is attached to the lateral aspect of the Temporo-Mandibular "
                "Joint (TMJ)."
            ),
            "why_others_incorrect": (
                "The ligament is not attached to the posterior aspect of TMJ, the mandibular condyle, or the "
                "coronoid process."
            ),
            "correct_text": "Lateral aspect of TMJ",
            "correct_index": 0,
        },
        "prompt_type": "default",
    },
    {
        "input": {
            "Question": "Lipoproteins are of how many types?",
            "Options": {"0": "3", "1": "2", "2": "4", "3": "None of the above"},
        },
        "Output": {
            "why_correct": (
                "Lipoproteins are classified into five main types: Chylomicrons, Very Low-Density Lipoproteins "
                "(VLDL), Intermediate-Density Lipoproteins (IDL), Low-Density Lipoproteins (LDL), and "
                "High-Density Lipoproteins (HDL)."
            ),
            "why_others_incorrect": (
                "Options '3', '2', and '4' are incorrect as they do not correctly represent the number of main "
                "lipoprotein types."
            ),
            "correct_text": "None of the above",
            "correct_index": 3,
        },
        "prompt_type": "task_specific",
    },
    {
        "input": {
            "Question": "Which vitamin is supplied from only animal source:",
            "Options": {
                "0": "Vitamin C",
                "1": "Vitamin B7",
                "2": "Vitamin B12",
                "3": "Vitamin D",
                "4": "Vitamin E",
            },
        },
        "Output": {
            "why_correct": "Vitamin B12 is primarily found in animal sources, including meats, dairy, and eggs.",
            "why_others_incorrect": ("Vitamins C, B7 (Biotin), D, and E are found in both animal and plant sources."),
            "correct_text": "Vitamin B12",
            "correct_index": 2,
        },
        "prompt_type": "default",
    },
    {
        "input": {
            "Question": "Parts of the brain that occur in epilepsy than aggressive behavior often appear?",
            "Options": {
                "0": "None of the above",
                "1": "Parietal",
                "2": "Occipital",
                "3": "Basal ganglia",
            },
        },
        "Output": {
            "why_correct": "No specific part of the brain is associated with aggression in epilepsy.",
            "why_others_incorrect": (
                "The parietal, occipital, and basal ganglia are not specifically implicated in aggression in epilepsy."
            ),
            "correct_text": "None of the above",
            "correct_index": 0,
        },
        "prompt_type": "task_specific",
    },
]
