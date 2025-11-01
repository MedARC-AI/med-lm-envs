# ACI-BENCH Medarc Eval: A Curated and Enriched Dataset for Clinical Note Generation

This repository contains `aci-bench-medarc-eval`, a consolidated, de-duplicated, and restructured version of the original [ACI-BENCH dataset](https://github.com/wyim/aci-bench/tree/main/data).

The original ACI-BENCH corpus is a valuable resource for benchmarking AI models that generate clinical notes from doctor-patient conversations. However, its complex structure—spread across dozens of files with varying ID formats and data versions—presents challenges for direct use in LLM training and evaluation.

This curated version addresses these problems by aggregating all available data into a single, unified file with an enriched, self-documenting structure.

## 1. About the Dataset

This dataset is designed for the task of **dialogue-to-note summarization**. It is a **comprehensive aggregation** of all data from the original ACI-BENCH repository. Each row contains:
*   A transcript of a doctor-patient encounter.
*   The corresponding ground-truth clinical note written by a medical professional.

The data has been meticulously processed to standardize column names and consolidate all source files, providing a complete and ready-to-use resource.

### New, Enriched Structure

To make the dataset more powerful and informative, it has been restructured from the original's simpler format. The final dataset contains the following five columns:

*   `id`: A standardized, unique identifier for the encounter.
*   `transcript`: The full text of the dialogue.
*   `note`: The ground-truth clinical note.
*   `transcript_listener`: Describes the **scenario** of the conversation (`aci`, `virtscribe`, or `virtassist`). This indicates *how* the doctor and patient were being listened to.
*   `transcript_writer`: Describes the **method of transcription** (`humantrans`, `asrcorr`, or `asr`). This indicates the *quality* and *origin* of the transcript.

This enriched structure allows users to perform nuanced experiments, such as evaluating an LLM's performance on natural conversations versus direct dictation, or on perfect transcripts versus error-prone ones, by simply filtering the `transcript_listener` and `transcript_writer` columns.

## 2. The Curation and Cleaning Process

The transformation from the original ACI-BENCH files to this clean dataset was a multi-step process of aggregation, investigation, and precise de-duplication.

### Initial Aggregation

The first step was to consolidate all data from the original's two folders (`src_experiment_data` and `challenge_data`). This initial aggregation produced a dataset of **464 total records**, including every available version of each transcript.

### The Duplication Anomaly

A deep analysis of the aggregated dataset revealed a significant data duplication issue.
*   The analysis found **77 distinct groups of duplicate records**.
*   Each group contained two rows with identical content (`transcript_listener`, `transcript_writer`, and `transcript`) but different ID formats.
*   The key pattern discovered was that one row had a standard ID, while the duplicate row had an ID that always started with the prefix **`D`**.

These "D" records were identified as true duplicates, likely artifacts from an earlier stage of the dataset's creation.

### The Final Solution

Based on this discovery, a final cleaning script was executed to perform a precise de-duplication:
1.  It identified all 77 duplicate groups.
2.  It systematically removed the **77 rows** associated with the IDs starting with "D".

## 3. The Final `aci-bench-medarc-eval` Dataset

This is the final, reliable, and validated dataset for any LLM experiment.

*   **Hugging Face Hub:** `harsh-c137/aci-bench-medarc-eval`
*   **Total Rows:** 387
*   **Integrity:** The dataset is confirmed to be free of duplicate encounters. The final row count reflects the comprehensive aggregation of all unique records present in the source files.