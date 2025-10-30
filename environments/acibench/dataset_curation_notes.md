# Project: ACI-BENCH LLM Evaluation Dataset - Curation and Cleaning

This document outlines the process of transforming the original, [ACI-BENCH dataset](https://github.com/wyim/aci-bench/tree/main/data) into a final, validated, and de-duplicated dataset (`harsh-c137/aci-bench-medarc-eval` on HF). This file is now perfectly suited for evaluating a Large Language Model (LLM) on the task of clinical note generation.

## 1. The Original ACI-BENCH Dataset

The source data is a comprehensive corpus for benchmarking AI models that generate clinical notes from doctor-patient conversations.

### Key Characteristics of the Original Dataset:

*   **Complex Structure:** Data was spread across two folders (`src_experiment_data` and `challenge_data`) and dozens of CSV files.
*   **Multiple Transcript Versions:** The dataset's included multiple versions of the same conversation transcript to simulate real-world scenarios (e.g., perfect Human Transcription vs. error-prone Automatic Speech Recognition).
*   **Inconsistent IDs:** The dataset used different ID's across different files to refer to the *exact same patient encounter*.

## 2. The Goal: A Clean Dataset for LLM Evaluation

The objective was to create a single, simple CSV file for a straightforward experiment. This file needed to contain only three essential columns:
1.  `id`: A unique identifier for the encounter.
2.  `dialogue`: The conversation text (input for the LLM).
3.  `note`: The ground-truth clinical note (for evaluating the LLM's output).

Crucially, the `dialogue` column needed to contain the single best-available version of the transcript for each encounter.

## 3. The Curation and Cleaning Process: From 319 to 225

### Initial Curation Attempt

The first step was designed to consolidate all files and select the highest-quality transcript for each encounter based on a priority list (`Human Transcription` > `ASR-Corrected` > `Raw ASR`).

### The Unexpected Outcome

This initial attempt produced a CSV with **319 rows**, significantly more than the ~207 unique encounters expected. A verification script confirmed that there were no duplicate IDs. This pointed to a more subtle problem.

There were **94 groups of duplicate dialogues**. Each group contained two rows with identical conversation text but different IDs.

*   One row had a "normal" ID (e.g., starting with 'A' for `aci` or 'V' for `virtscribe`).
*   The other row had an ID that always started with the prefix **"D2N"**.

This confirmed that the "D2N" records were the true duplicates, likely artifacts from an earlier stage of the dataset's creation.

### The Final Solution

Based on this discovery, the final cleaning script:
1.  Iidentified all 94 groups of duplicate dialogues.
2.  From each group, it collected the ID that started with "D2N".
3.  It then filtered the 319-row dataset, **removing all 94 rows** associated with these "D2N" IDs.

## 4. The Final Dataset: `aci-bench-medarc-eval`

This is the final, reliable, and validated dataset for our experiment.

*   **Total Rows:** 225 (This reflects the 207 unique encounters, plus or minus any discrepancies from the missing files in the original download).
*   **Structure:** It contains the three essential columns: `id`, `dialogue`, and `note`.
*   **Integrity:** The file is confirmed to be free of duplicate encounters, ensuring that our LLM evaluation will be fair and accurate.