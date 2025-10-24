import os
import re
import json
import warnings
from pathlib import Path
from typing import List
from collections import OrderedDict

import verifiers as vf
from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate


warnings.filterwarnings("ignore", category=UserWarning, module='evaluate')


subjective_subsections = OrderedDict()
subjective_subsections['cc'] = ['cc :', 'chief complaint :', 'reason for visit :', "CHIEF COMPLAINT"]
subjective_subsections['hpi'] = ['history :', 'history of present illness :', 'history of present illness', 'hpi :', 'hpi', 'hpi notes :', 'interval history :', 'interval hx :', 'subjective :', "HISTORY OF PRESENT ILLNESS"]
subjective_subsections['ros'] = ['ros :', 'review of system :', 'review of systems :']
subjective_subsections['other_histories'] = ['SOCIAL HISTORY', "PAST HISTORY"]
objectiveexam_subsections = OrderedDict()
objectiveexam_subsections['pe'] = ['physical exam :', 'physical examination :', 'pe :', 'physical findings :', 'examination :', 'exam :', "PHYSICAL EXAMINATION", "PHYSICAL EXAM", ]
objectiveexam_subsections['vitals'] = ['VITALS REVIEWED', ]
objectiveresults_subsections = OrderedDict()
objectiveresults_subsections['findings'] = ['results :', 'findings :', "RESULTS", ]
ap_subsections = OrderedDict()
ap_subsections['assessment'] = ['assessment :', 'a:']
ap_subsections['plan'] = ['plan :', 'plan of care :', 'p:', 'medical decision-making plan :', 'summary plan']
ap_subsections['ap'] = ['ap :', 'a / p :', 'assessment and plan :', 'assessment & plan :', 'disposition / plan :', "ASSESSMENT AND PLAN", ]
sectcat2subsections = OrderedDict()
sectcat2subsections['subjective'] = subjective_subsections
sectcat2subsections['objective_exam'] = objectiveexam_subsections
sectcat2subsections['objective_results'] = objectiveresults_subsections
sectcat2subsections['assessment_and_plan'] = ap_subsections
NOSECTIONHEADER = 'default'
subsectionheader2section = {}
for sh, sshdicts in sectcat2subsections.items():
    for ssh, lst in sshdicts.items():
        subsectionheader2section[ssh] = sh
subsectionheader2section[NOSECTIONHEADER] = NOSECTIONHEADER

class SectionTagger():
    def __init__(self, sectcat2subsections=sectcat2subsections):
        self.sectcat2subsections = sectcat2subsections
        self.compileregexes()
    def compileregexes(self):
        self.subsect2regex = {}
        for _, sectcat2subsections in self.sectcat2subsections.items():
            for subsect, vlst in sectcat2subsections.items():
                self.subsect2regex[subsect] = self._compile_regexexpression(vlst)
    def _compile_regexexpression(self, vlst):
        expressions, otherexps = [], []
        for exp in vlst:
            exp2 = '(' + re.escape(exp).replace(r'\ ', r'\s*') + ')'
            expressions.append(exp2)
            if exp.endswith(':'):
                exp2 = '(' + re.escape(exp[:-1]).replace(r'\ ', r'\s*') + ')'
                otherexps.append(exp2)
        patt = r'\s*(?P<sectionheader1>' + '|'.join(expressions) + ').*'
        if otherexps:
            pattott = r'\s*(?P<sectionheader2>' + '|'.join(otherexps) + r')\s*$'
            return f'({patt}|{pattott})'
        return patt
    def tag_sections(self, text):
        subsects = []
        offset = 0
        for linenum, line in enumerate(text.split('\n')):
            for subsect, rgx in self.subsect2regex.items():
                m = re.match(rgx, line, re.IGNORECASE)
                if m:
                    secthlib = m.groupdict()
                    secthpattname = 'sectionheader1' if secthlib.get('sectionheader1') else 'sectionheader2'
                    subsects.append((subsect, linenum, offset, offset + m.end(secthpattname)))
            offset += len(line) + 1
        linenum2tuple = {s[1]: s for s in subsects}
        sectionlist = []
        prevsectionheadertuple = linenum2tuple.get(0, (NOSECTIONHEADER, 0, 0, 0))
        lines = text.split('\n')
        offset = len(lines[0]) + 1
        for linenum, line in enumerate(lines[1:], 1):
            if linenum in linenum2tuple:
                shtuple = linenum2tuple[linenum]
                prevsection = subsectionheader2section[prevsectionheadertuple[0]]
                sectionlist.append([prevsection] + list(prevsectionheadertuple) + [offset])
                prevsectionheadertuple = shtuple
            offset += len(line) + 1
        prevsection = subsectionheader2section[prevsectionheadertuple[0]]
        sectionlist.append([prevsection] + list(prevsectionheadertuple) + [len(text)])
        return sectionlist
    def divide_note_by_metasections(self, text):
        detected_sections = self.tag_sections(text)
        if detected_sections and detected_sections[0][0] == NOSECTIONHEADER:
            detected_sections[0][0] = 'subjective'
        meta_sections_map = {'subjective': None, 'objective_exam': None, 'objective_results': None, 'assessment_and_plan': None}
        for section in detected_sections:
            if section[0] in meta_sections_map and meta_sections_map[section[0]] is None:
                meta_sections_map[section[0]] = section
        meta_sections = [s for s in meta_sections_map.values() if s is not None]
        meta_sections.sort(key=lambda x: x[3])
        for i in range(len(meta_sections) - 1):
            meta_sections[i][-1] = meta_sections[i + 1][-3]
        if meta_sections:
            meta_sections[-1][-1] = len(text)
        return meta_sections


SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
def add_section_divisions(row, section_tagger):
    row['src_len'] = len(row['dialogue'].split())
    for eval_type in ['reference', 'prediction']:
        text = str(row[eval_type]).replace('__lf1__', '\n')
        detected_divisions = section_tagger.divide_note_by_metasections(text)
        for label, _, _, start, _, end in detected_divisions:
            row[f'{eval_type}_{label}'] = text[start:end].replace('\n', '__lf1__')
    return row

def calculate_all_metrics(prediction_data, section_tagger):
    df = pd.DataFrame(prediction_data)
    df.rename(columns={'ground_truth': 'reference'}, inplace=True)
    if 'dataset' not in df.columns: df['dataset'] = 'default_dataset'
    print("Applying section divisions to notes...")
    df = df.apply(lambda row: add_section_divisions(row, section_tagger), axis=1)
    df = df.fillna('')
    num_test = len(df)
    references, predictions = df['reference'].tolist(), df['prediction'].tolist()
    for division in SECTION_DIVISIONS:
        references.extend(df.get(f'reference_{division}', [''] * num_test).tolist())
        predictions.extend(df.get(f'prediction_{division}', [''] * num_test).tolist())
    print("Calculating metrics (ROUGE, BERTScore, BLEURT)...")
    results_rouge_all = evaluate.load('rouge').compute(references=references, predictions=predictions, use_aggregator=False)
    results_bertscore = evaluate.load('bertscore').compute(references=references, predictions=predictions, model_type='microsoft/deberta-xlarge-mnli', device="cuda")
    results_bleurt = evaluate.load('bleurt', config_name='BLEURT-20').compute(references=references, predictions=predictions)
    print("Aggregating results...")
    results_all = {"num_test": num_test, 'ALL': {
        'rouge1': np.mean(results_rouge_all['rouge1'][:num_test]),
        'rouge2': np.mean(results_rouge_all['rouge2'][:num_test]),
        'rougeL': np.mean(results_rouge_all['rougeL'][:num_test]),
        'bertscore-f1': np.mean(results_bertscore['f1'][:num_test]),
        'bleurt': np.mean(results_bleurt['scores'][:num_test]),
    }}
    for i, division in enumerate(SECTION_DIVISIONS):
        start, end = (i + 1) * num_test, (i + 2) * num_test
        results_all[f'division-{division}'] = {
            'rouge1': np.mean(results_rouge_all['rouge1'][start:end]),
            'bertscore-f1': np.mean(results_bertscore['f1'][start:end]),
        }
    print("Metric calculation complete.")
    return results_all


def load_environment(
    repo_id: str = "harsh-c137/aci-bench-basic",
    split: str = "test",
):
    return ACIBenchEnv(repo_id=repo_id, split=split)

class ACIBenchEnv(vf.Environment):
    def __init__(self, repo_id: str, split: str):
        super().__init__()
        print(f"Loading ACI-Bench dataset from Hugging Face repo: {repo_id}, split: {split}")
        try:
            self.dataset = load_dataset(repo_id, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{repo_id}' with split '{split}'. Error: {e}")
        
        print("Initializing SectionTagger...")
        self.section_tagger = SectionTagger()
        print("Initialization complete.")

    def rollout(self, client, model: str, *, num_examples: int, **kwargs) -> vf.Rollout:
        print(f"Generating summaries for {num_examples} examples using model: {model}...")
        results: List[vf.Result] = []
        dataset_subset = self.dataset.select(range(num_examples))
        for item in dataset_subset:
            dialogue, ground_truth = item["dialogue"], item["note"]
            prompt = f"Please provide a clinical summary for the following dialogue:\n\n{dialogue}"
            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            prediction = response.choices[0].message.content.strip()
            results.append(vf.Result(prompt=prompt, completion=prediction, info={"ground_truth": ground_truth, "dialogue": dialogue}))
        return vf.Rollout(results=results)

    def evaluate_rollout(self, rollout: vf.Rollout) -> vf.Rollout:
        print("Rollout complete. Starting batch metric calculation...")
        prediction_data = [{"prediction": r.completion, "ground_truth": r.info["ground_truth"], "dialogue": r.info["dialogue"]} for r in rollout.results]
        final_scores = calculate_all_metrics(prediction_data, self.section_tagger)
        rollout.info["aggregated_scores"] = final_scores
        print("\n--- Aggregated ACI-Bench Scores ---")
        print(json.dumps(final_scores, indent=4))
        print("-----------------------------------")
        return rollout