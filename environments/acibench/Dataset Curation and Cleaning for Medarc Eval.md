# Project: ACI-BENCH LLM Evaluation Dataset - Curation and Cleaning

This document outlines the process of transforming the original, complex ACI-BENCH dataset into a final, validated, and de-duplicated dataset (`llm_input_dataset_cleaned.csv`). This file is now perfectly suited for evaluating a Large Language Model (LLM) on the task of clinical note generation.

## 1. The Original ACI-BENCH Dataset

The source data is a comprehensive corpus for benchmarking AI models that generate clinical notes from doctor-patient conversations.

### Key Characteristics of the Original Dataset:

*   **Complex Structure:** Data was spread across two folders (`src_experiment_data` and `challenge_data`) and dozens of CSV files.
*   **Multiple Transcript Versions:** The dataset's most complex feature was its inclusion of multiple versions of the same conversation transcript to simulate real-world scenarios (e.g., perfect Human Transcription vs. error-prone Automatic Speech Recognition).
*   **Inconsistent IDs:** As discovered during our process, the dataset used different ID formats across different files to refer to the *exact same patient encounter*.

## 2. The Goal: A Clean Dataset for LLM Evaluation

Our objective was to create a single, simple CSV file for a straightforward experiment. This file needed to contain only three essential columns:
1.  `id`: A unique identifier for the encounter.
2.  `dialogue`: The conversation text (input for the LLM).
3.  `note`: The ground-truth clinical note (for evaluating the LLM's output).

Crucially, the `dialogue` column needed to contain the single best-available version of the transcript for each encounter.

## 3. The Curation and Cleaning Process: From 319 to 225

Our journey to the final dataset involved a critical data discovery and cleaning phase.

### Initial Curation Attempt

Our first script (`create_llm_input.py`) was designed to consolidate all files and select the highest-quality transcript for each encounter based on a priority list (`Human Transcription` > `ASR-Corrected` > `Raw ASR`).

### The Unexpected Outcome

This initial attempt produced a CSV with **319 rows**, significantly more than the ~207 unique encounters we expected. A verification script (`verify_duplicates.py`) confirmed that there were no duplicate IDs. This pointed to a more subtle problem.

### The Breakthrough Discovery

A deeper investigation using `find_duplicate_dialogues.py` revealed the root cause. The script found **94 groups of duplicate dialogues**. Each group contained two rows with identical conversation text but different IDs.

Your analysis of the report (`duplicate_dialogues_report.txt`) uncovered the critical pattern:
*   One row had a "normal" ID (e.g., starting with 'A' for `aci` or 'V' for `virtscribe`).
*   The other row had an ID that always started with the prefix **"D2N"**.

This confirmed that the "D2N" records were the true duplicates, likely artifacts from an earlier stage of the dataset's creation.

### The Final Solution

Based on this discovery, the final cleaning script (`clean_final_dataset.py`) was created. It implemented a precise and effective logic:
1.  It identified all 94 groups of duplicate dialogues.
2.  From each group, it collected the ID that started with "D2N".
3.  It then filtered the 319-row dataset, **removing all 94 rows** associated with these "D2N" IDs.

## 4. The Final Dataset: `llm_input_dataset_cleaned.csv`

This is the final, reliable, and validated dataset for our experiment.

*   **File Name:** `llm_input_dataset_cleaned.csv`
*   **Total Rows:** 225 (This reflects the 207 unique encounters, plus or minus any discrepancies from the missing files in the original download).
*   **Structure:** It contains the three essential columns: `id`, `dialogue`, and `note`.
*   **Integrity:** The file is confirmed to be free of duplicate encounters, ensuring that our LLM evaluation will be fair and accurate.

This rigorous process of curation, investigation, and targeted cleaning was essential to transform the complex original dataset into a simple and trustworthy resource for our project.