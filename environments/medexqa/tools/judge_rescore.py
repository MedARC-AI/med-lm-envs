import argparse
import asyncio
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI

# Reuse existing judge implementations from the environment
from environments.medexqa.geval_judge.geval_judge import (
    explanation_geval_reward as geval_reward,
)
from environments.medexqa.factscore_judge.atomic_facts_judge import (
    explanation_factscore_reward as factscore_reward,
)


def _extract_numeric(text: str) -> float:
    m = re.search(r"(\d+\.\d+|\d+)", (text or "").strip())
    if not m:
        return 0.0
    try:
        val = float(m.group(1))
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.0


def _read_results(paths: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for p in paths:
        try:
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rows.append((p, rec))
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return rows


class JudgeRecorder:
    def __init__(self, client: AsyncOpenAI, model: str, sleep_ms: int = 500, max_retries: int = 5, verbose: bool = True, max_tokens: int = 384):
        self.client = client
        self.model = model
        self.sleep_ms = max(0, int(sleep_ms))
        self.max_retries = max(1, int(max_retries))
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.logs: List[Dict[str, str]] = []

    async def __call__(self, messages, *_args, **_kwargs) -> str:
        # messages is a list of {role, content}
        content = messages[-1].get("content", "") if messages else ""
        attempt = 0
        delay = self.sleep_ms / 1000.0
        while True:
            try:
                if self.verbose:
                    print(f"[judge] calling model={self.model}, tokens<=256")
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
                out = resp.choices[0].message.content or ""
                self.logs.append({"prompt": content, "response": out})
                # throttle between calls
                if self.sleep_ms > 0:
                    await asyncio.sleep(self.sleep_ms / 1000.0)
                return out
            except Exception as e:
                attempt += 1
                msg = str(e)
                if self.verbose:
                    print(f"[judge] error on attempt {attempt}: {msg}")
                # retry on rate limit or transient errors
                if attempt < self.max_retries:
                    # exponential backoff with floor at configured delay
                    backoff = delay * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                # record failure
                self.logs.append({"prompt": content, "response": f"<ERROR>: {msg}"})
                return ""


async def judge_geval(
    client: AsyncOpenAI,
    model: str,
    rec: Dict[str, Any],
    *,
    sleep_ms: int = 500,
    max_retries: int = 5,
    verbose: bool = True,
    max_tokens: int = 384,
) -> Tuple[float, str, str]:
    info = rec.get("info", {}) or {}
    question = info.get("question", "")
    options = {k: info.get(k, "") for k in ["A", "B", "C", "D"]}
    exp0 = info.get("exp0", "")
    exp1 = info.get("exp1", "")
    answer = rec.get("answer", "")
    completion_msgs = rec.get("completion", [])

    jr = JudgeRecorder(client, model, sleep_ms=sleep_ms, max_retries=max_retries, verbose=verbose, max_tokens=max_tokens)
    score = await geval_reward(jr, None, completion_msgs, answer, state={}, info=info, judge_client=client, judge_model=model)
    # Last log entry contains the overall prompt/response
    judge_output = jr.logs[-1]["response"] if jr.logs else ""
    refs = (
        f"Question: {question}\n"
        f"Correct answer: {answer} ({options.get(answer,'')})\n"
        f"Ref1: {exp0}\n"
        f"Ref2: {exp1}"
    )
    return float(score), judge_output, refs


async def judge_factscore(
    client: AsyncOpenAI,
    model: str,
    rec: Dict[str, Any],
    *,
    sleep_ms: int = 500,
    max_retries: int = 5,
    verbose: bool = True,
    max_tokens: int = 384,
    use_coverage: bool = False,
) -> Tuple[float, str, str]:
    info = rec.get("info", {}) or {}
    question = info.get("question", "")
    options = {k: info.get(k, "") for k in ["A", "B", "C", "D"]}
    exp0 = info.get("exp0", "")
    exp1 = info.get("exp1", "")
    answer = rec.get("answer", "")
    completion_msgs = rec.get("completion", [])

    jr = JudgeRecorder(client, model, sleep_ms=sleep_ms, max_retries=max_retries, verbose=verbose, max_tokens=max_tokens)
    score = await factscore_reward(jr, None, completion_msgs, answer, state={}, info=info, judge_client=client, judge_model=model, use_coverage=use_coverage)

    # Parse logs to reconstruct claim labels and track extraction outcomes
    labels: List[Tuple[str, str]] = []  # (claim, label) where passage=references (support)
    coverage_labels: List[Tuple[str, str]] = []  # (ref_claim, label) where passage=explanation (coverage)
    extraction_responses: List[str] = []
    for entry in jr.logs:
        prompt = entry.get("prompt", "") or ""
        response = entry.get("response", "") or ""

        # Look for new format: "CLAIM TO VERIFY:\n<claim>"
        m = re.search(r"CLAIM TO VERIFY:\s*\n(.+?)(?:\n\nINSTRUCTIONS:)", prompt, flags=re.DOTALL)
        if m:
            claim = m.group(1).strip()
            # Heuristic: prompts containing "PASSAGE (Combined References):" belong to support
            # Prompts with "MODEL EXPLANATION:" belong to coverage
            if "PASSAGE (Combined References):" in prompt:
                labels.append((claim, response.strip().upper()))
            elif "MODEL EXPLANATION:" in prompt:
                coverage_labels.append((claim, response.strip().upper()))

        # Fallback: old format "Fact:\n<fact>"
        if not m:
            m_old = re.search(r"Fact:\s*\n(.+)$", prompt, flags=re.DOTALL)
            if m_old:
                claim = m_old.group(1).strip()
                if "Question:" in prompt:
                    labels.append((claim, response.strip().upper()))
                else:
                    coverage_labels.append((claim, response.strip().upper()))

        if "Claims JSON:" in prompt:
            extraction_responses.append(response)

    refs = (
        f"Question: {question}\n"
        f"Correct: ({answer}) {options.get(answer,'')}\n"
        f"Ref1: {exp0}\n"
        f"Ref2: {exp1}"
    )
    # Derive error tag for extraction phase
    err_tag = ""
    if extraction_responses:
        last_extraction = extraction_responses[-1]
        try:
            parsed = json.loads(last_extraction)
            if isinstance(parsed, list) and len(parsed) == 0:
                err_tag = "empty_extraction"
        except Exception:
            err_tag = "extraction_error"
    elif not labels and not coverage_labels:
        err_tag = "empty_extraction"

    # Compute support/coverage rates from labels (handle 3-level format)
    def _rate(pairs: List[Tuple[str, str]]) -> float:
        if not pairs:
            return 0.0
        total_score = 0.0
        for _, lbl in pairs:
            lbl_clean = (lbl or "").strip().upper()
            if "FULLY_SUPPORTED" in lbl_clean or "FULLY SUPPORTED" in lbl_clean:
                total_score += 1.0
            elif "PARTIALLY_SUPPORTED" in lbl_clean or "PARTIALLY SUPPORTED" in lbl_clean:
                total_score += 0.5
            elif lbl_clean.startswith("TRUE"):  # Fallback for old format
                total_score += 1.0
            # NOT_SUPPORTED or FALSE = 0.0
        return float(total_score) / float(len(pairs))

    support_rate = _rate(labels)
    coverage_rate = _rate(coverage_labels)

    details = json.dumps({
        "claims": labels,
        "coverage_labels": coverage_labels,
        "support_rate": support_rate,
        "coverage_rate": coverage_rate,
    }, ensure_ascii=False)
    if err_tag:
        refs = refs + f"\n<ERR>: {err_tag}"
    return float(score), details, refs


async def main():
    ap = argparse.ArgumentParser(description="Re-score saved MedExQA completions with LLM judges.")
    ap.add_argument("--base", default="https://openrouter.ai/api/v1", help="Judge API base URL")
    ap.add_argument("--model", default="openai/gpt-oss-20b:free", help="Judge model id")
    ap.add_argument("--key_var", default="OPENAI_API_KEY", help="Env var name holding the API key")
    ap.add_argument("--input_glob", default="environments/medexqa/outputs/evals/**/results.jsonl", help="Glob to results.jsonl files")
    ap.add_argument("--out_csv_prefix", default="environments/medexqa/outputs/judge_scores/medexqa_", help="Output CSV prefix (will append judge name)")
    ap.add_argument("--sleep_ms", type=int, default=500, help="Sleep/throttle between judge calls (ms)")
    ap.add_argument("--max_retries", type=int, default=5, help="Max retries on judge call errors")
    ap.add_argument("--max_tokens", type=int, default=384, help="Max tokens per judge response")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--judge", choices=["geval", "factscore", "both"], default="both", help="Which judge(s) to run")
    ap.add_argument("--use_coverage", action="store_true", help="Enable coverage calculation for FactScore (slower but more comprehensive)")
    args = ap.parse_args()

    api_key = os.getenv(args.key_var)
    if not api_key:
        raise SystemExit(f"Missing API key in env var {args.key_var}")

    client = AsyncOpenAI(base_url=args.base, api_key=api_key)

    # Discover saved runs
    paths = sorted(glob.glob(args.input_glob, recursive=True))
    if args.verbose:
        print(f"Scanning {len(paths)} results.jsonl files...")
    rows = _read_results(paths)
    if not rows:
        print("No results found to re-score.")
        return

    os.makedirs(os.path.dirname(args.out_csv_prefix), exist_ok=True)

    # Prepare CSV writers conditionally
    gwriter = None
    fwriter = None
    geval_path = args.out_csv_prefix + "geval.csv"
    fact_path = args.out_csv_prefix + "factscore.csv"
    if args.judge in ("geval", "both"):
        gf = open(geval_path, "w", newline="")
        gwriter = csv.writer(gf)
        gwriter.writerow(["run_file", "specialty", "question", "A", "B", "C", "D", "answer", "completion", "judge_model_output", "judge_score", "references", "error"])
    if args.judge in ("factscore", "both"):
        ff = open(fact_path, "w", newline="")
        fwriter = csv.writer(ff)
        fwriter.writerow(["run_file", "specialty", "question", "A", "B", "C", "D", "answer", "completion", "claims_labels_json", "support_rate", "coverage_labels_json", "coverage_rate", "final_score", "references", "error"]) 

    # Process sequentially to keep it simple
    for idx, (run_file, rec) in enumerate(rows, start=1):
        info = rec.get("info", {}) or {}
        spec = info.get("specialty", "")
        question = info.get("question", "")
        A = info.get("A", "")
        B = info.get("B", "")
        C = info.get("C", "")
        D = info.get("D", "")
        answer = rec.get("answer", "")
        completion_msgs = rec.get("completion", [])
        completion_text = completion_msgs[-1].get("content", "") if completion_msgs else ""

        if args.verbose:
            print(f"[{idx}/{len(rows)}] {run_file} | spec={spec} | len(prompt)={len(question)} | len(completion)={len(completion_text)}")

        # G-Eval
        if args.judge in ("geval", "both") and gwriter is not None:
            if args.verbose:
                print("  -> G-Eval judging...")
            g_score, g_out, g_refs = await judge_geval(
                client,
                args.model,
                rec,
                sleep_ms=args.sleep_ms,
                max_retries=args.max_retries,
                verbose=args.verbose,
                max_tokens=args.max_tokens,
            )
            # detect errors in logs
            g_err = ""
            if g_out.strip().startswith("<ERROR>") or g_out.strip() == "":
                g_err = "empty_or_error"
            gwriter.writerow([run_file, spec, question, A, B, C, D, answer, completion_text, g_out, f"{g_score:.3f}", g_refs, g_err])

        # FactScore
        if args.judge in ("factscore", "both") and fwriter is not None:
            if args.verbose:
                print("  -> FactScore judging...")
            f_score, f_details, f_refs = await judge_factscore(
                client,
                args.model,
                rec,
                sleep_ms=args.sleep_ms,
                max_retries=args.max_retries,
                verbose=args.verbose,
                max_tokens=args.max_tokens,
                use_coverage=args.use_coverage,
            )
            f_err = ""
            support_rate = ""
            coverage_rate = ""
            coverage_labels_json = "{}"
            try:
                dd = json.loads(f_details)
                support_rate = f"{float(dd.get('support_rate', 0.0)):.3f}"
                coverage_rate = f"{float(dd.get('coverage_rate', 0.0)):.3f}"
                coverage_labels_json = json.dumps(dd.get("coverage_labels", []), ensure_ascii=False)
            except Exception:
                pass
            if f_details.strip() == "" or f_details.strip() == "{}":
                f_err = "empty_or_error"
            fwriter.writerow([run_file, spec, question, A, B, C, D, answer, completion_text, f_details, support_rate, coverage_labels_json, coverage_rate, f"{f_score:.3f}", f_refs, f_err])

    if gwriter is not None:
        gf.close()
        print(f"Wrote: {geval_path}")
    if fwriter is not None:
        ff.close()
        print(f"Wrote: {fact_path}")


if __name__ == "__main__":
    asyncio.run(main())


