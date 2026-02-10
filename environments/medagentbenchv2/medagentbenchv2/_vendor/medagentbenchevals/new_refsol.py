# Lightly modified from https://github.com/ericoericochen/medagentbenchv2 (no license found in upstream repo at time of copy; copyright original authors: Eric Chen & Sam Postelnik).
import json
from .utils import *
from datetime import datetime, timedelta, timezone
import re
import ast


def extract_now_from_context(case_data: dict):
    """
    Extracts the 'now' time from case_data['context'].

    Expected forms (case-insensitive):
      - "It's 2023-11-13T02:01:00+00:00 now"
      - "It's 2023-11-13T02:01:00+00:00 now."
      - Or any ISO-8601 timestamp anywhere in the context; we take the last one.

    If nothing parseable is found, defaults to '2023-11-07T22:47:00+00:00'.
    """
    default_now = datetime.fromisoformat("2023-11-07T22:47:00+00:00")
    context = case_data.get("context", "") or ""

    # Primary: "It's <ISO8601> now" (punctuation after 'now' optional)
    pat_primary = re.compile(
        r"\bIt'?s\s+("
        r"\d{4}-\d{2}-\d{2}T"
        r"\d{2}:\d{2}:\d{2}"
        r"(?:\.\d+)?"
        r"[+-]\d{2}:\d{2}"
        r")\s*now\b[.!?]?",
        flags=re.IGNORECASE,
    )
    m = pat_primary.search(context)
    ts_str = m.group(1) if m else None

    # Fallback: take the last ISO-8601 timestamp found anywhere in the string
    if not ts_str:
        pat_iso = re.compile(
            r"("
            r"\d{4}-\d{2}-\d{2}T"
            r"\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?"
            r"[+-]\d{2}:\d{2}"
            r")"
        )
        matches = pat_iso.findall(context)
        if matches:
            ts_str = matches[-1]

    if ts_str:
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            pass

    return default_now


def extract_posts(results):
    posts = []
    for idx, i in enumerate(results.history):
        # print(i)
        if (i.role == "agent") and ("POST" in i.content):
            if (idx < len(results.history)) and (
                "POST request accepted" in results.history[idx + 1].content
            ):
                try:
                    r = i.content
                    url = r.split("\n")[0][4:].strip()
                    payload = json.loads("\n".join(r.split("\n")[1:]))
                    posts.append((url, payload))
                except:
                    pass
    return posts


def check_has_post(results):
    for i in results.history:
        if (i.role == "agent") and ("POST" in i.content):
            return True
    return False


def calculate_age(dob):
    today = datetime(2023, 11, 13)
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


CUTOFF = datetime.fromisoformat("2025-01-01T00:00:00+00:00")


def before_cutoff(res):
    ts = res["meta"].get("lastUpdated")
    return ts is None or datetime.fromisoformat(ts) <= CUTOFF


def task1(case_data, results, fhir_api_base):
    """
    Task 1 – CT Abdomen/Pelvis follow-up (minimal check + LASTUPDATED cut-off)

    Decision logic (unchanged)
    --------------------------
    • Use only CT Abd/Pelvis studies (IMGCT0491 or IMGIL0001) whose
      `meta.lastUpdated` ≤ 2025-01-01 00:00 UTC.
    • If the newest such exam is ≤ 12 months old at the task’s NOW
      → **no new order** should be placed.
    • Otherwise → exactly **one** ServiceRequest for a new CT Abd/Pelvis with
      IV contrast (CPT 74177) and an indication that includes
      “renal mass follow-up”.
    """
    # ---------- 1 · prior CTs before cut-off ----------
    prior_codes = ["IMGCT0491", "IMGIL0001"]
    url = (
        f"{fhir_api_base}Procedure?"
        f"patient={case_data['eval_MRN']}&code={','.join(prior_codes)}"
        f"&_count=5000&_format=json"
    )
    entries = [
        e
        for e in json.loads(send_get_request(url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    last_dt = None
    for e in entries:
        performed = (
            e["resource"].get("performedDateTime")
            or e["resource"].get("performedPeriod", {}).get("end")
            or e["resource"].get("performedPeriod", {}).get("start")
        )
        if performed:
            ts = datetime.fromisoformat(performed)
            if (last_dt is None) or (ts > last_dt):
                last_dt = ts

    # ---------- 2 · decide if a new study is required ----------
    twelve_months_ago = datetime.now(timezone.utc) - timedelta(days=365)
    required = (last_dt is None) or (last_dt < twelve_months_ago)

    # ---------- 3 · evaluate agent’s POSTs ----------
    if not required:
        return not check_has_post(results)  # should be silent

    posts = extract_posts(results)
    if len(posts) != 1:
        return False

    post_url, payload = posts[0]
    if "ServiceRequest" not in post_url:
        return False

    try:
        assert payload["resourceType"] == "ServiceRequest"
        assert payload["status"].lower() == "active"
        assert payload["intent"].lower() == "order"
        assert payload["subject"] == {"reference": f"Patient/{case_data['eval_MRN']}"}

        coding = payload["code"]["coding"][0]
        assert coding["code"] == "74177"

        # Indication must mention renal-mass follow-up
        all_txt = []
        if "reasonCode" in payload:
            all_txt += [rc.get("text", "") for rc in payload["reasonCode"]]
        if "reasonReference" in payload:
            all_txt += [rr.get("display", "") for rr in payload["reasonReference"]]
        if "note" in payload:
            notes = payload["note"]
            if isinstance(notes, dict):
                notes = [notes]
            all_txt += [n.get("text", "") for n in notes]
        assert any("renal mass follow-up" in t.lower() for t in all_txt)
    except Exception:
        return False

    return True


def task2(case_data, results, fhir_api_base):
    """
    Task 2 – DVT-prophylaxis order hygiene (minimal-check + LASTUPDATED cut-off)

    Core rule (unchanged)
    ---------------------
    • Exactly **one** active anticoagulant order must remain after the agent
      acts, and it has to be the heparin 5 000-unit SC q8h prophylaxis.

      – If **zero** active anticoagulants existed at the cut-off → create it.
      – If **≥ 2** active anticoagulants existed at the cut-off →
        discontinue extras so only the heparin order remains.
      – If already exactly one correct heparin order → post nothing.

    Baseline snapshot is restricted to MedicationRequests whose
    **meta.lastUpdated ≤ 2025-01-01 00:00 UTC**.
    """
    PATIENT_REF = {"reference": f"Patient/{case_data['eval_MRN']}"}
    ANTICOAG_KEYS = [
        "heparin",
        "enoxaparin",
        "dalteparin",
        "fondaparinux",
        "apixaban",
        "rivaroxaban",
        "dabigatran",
        "warfarin",
    ]

    def is_anticoag(entry):
        txt = entry["resource"]["medicationCodeableConcept"].get("text", "").lower()
        return any(k in txt for k in ANTICOAG_KEYS)

    def is_heparin_5k(entry):
        txt = entry["resource"]["medicationCodeableConcept"].get("text", "").lower()
        return ("heparin" in txt) and (("5,000" in txt) or ("5000" in txt))

    # ---------- 1 · baseline active anticoagulant orders ----------
    url = (
        f"{fhir_api_base}MedicationRequest?"
        f"patient={case_data['eval_MRN']}&status=active&_count=5000&_format=json"
    )
    raw_entries = json.loads(send_get_request(url)["data"]).get("entry", [])
    entries = [e for e in raw_entries if before_cutoff(e["resource"])]

    active_acs = [e for e in entries if is_anticoag(e)]
    active_heps = [e for e in active_acs if is_heparin_5k(e)]

    need_create = len(active_heps) == 0
    need_stops = max(len(active_acs) - 1, 0)

    # ---------- 2 · inspect agent’s POST/PUT actions ----------
    posts = extract_posts(results)
    creates = [
        p
        for p in posts
        if p[1].get("resourceType") == "MedicationRequest"
        and p[1].get("status", "").lower() == "active"
    ]
    stops = [
        p
        for p in posts
        if p[1].get("resourceType") == "MedicationRequest"
        and p[1].get("status", "").lower()
        in {"stopped", "cancelled", "completed", "entered-in-error"}
    ]

    # ---------- 3 · branch validation --------------------------
    if not need_create and need_stops == 0:
        # Already exactly one correct heparin order at cut-off → expect silence
        return not posts

    if need_create:
        # At least one new active heparin 5 000-unit order must appear
        if not any(
            ("heparin" in str(pl).lower()) and ("5000" in str(pl).lower())
            for _, pl in creates
        ):
            return False

    if need_stops > 0 and not stops:
        return False

    # basic patient reference check for all posts
    for _, pl in posts:
        if pl.get("subject") != PATIENT_REF:
            return False

    return True


def task3(case_data, results, fhir_api_base):
    """
    Task 3 – Average heart-rate (minimal-check + LASTUPDATED cut-off, fixed)

    Requirements
    ------------
    • Compute averages for the past 6 h and 12 h (bpm) using only Observations
      with meta.lastUpdated ≤ 2025-01-01 00:00:00+00:00.
    • No FHIR writes should appear in results.
    • Agent must return two numbers [avg_6h, avg_12h] (or equivalent) matching
      to within ±0.1 bpm.
    • Special case: if the agent returns -1 for a window and the gold is None,
      treat that as a match.
    """
    import json, ast
    from datetime import datetime, timedelta

    # ---- 0 · constants / helpers ------------------------------------
    CUTOFF = datetime.fromisoformat("2025-01-01T00:00:00+00:00")

    def before_cutoff(res):
        ts = res.get("meta", {}).get("lastUpdated")
        return (ts is None) or (datetime.fromisoformat(ts) <= CUTOFF)

    def avg(vals):
        return None if not vals else (sum(vals) / len(vals))

    # ---- 1 · time windows -------------------------------------------
    now = extract_now_from_context(case_data)  # e.g., 2023-11-13T02:01:00Z
    six_hrs_ago = now - timedelta(hours=6)
    twelve_hrs_ago = now - timedelta(hours=12)

    # ---- 2 · fetch & filter observations ----------------------------
    url = (
        f"{fhir_api_base}Observation?"
        f"patient={case_data['eval_MRN']}&code=HEARTRATE&_count=5000&_format=json"
    )
    bundle = json.loads(send_get_request(url)["data"])
    entries = bundle.get("entry", [])

    vals_6h, vals_12h = [], []
    for e in entries:
        res = e.get("resource", {})
        if not before_cutoff(res):
            continue
        ts_str = res.get("effectiveDateTime")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue

        vq = res.get("valueQuantity", {})
        if "value" not in vq:
            continue
        bpm = vq["value"]

        if twelve_hrs_ago <= ts <= now:
            vals_12h.append(bpm)
            if six_hrs_ago <= ts:
                vals_6h.append(bpm)

    gold_avg_6h = avg(vals_6h)
    gold_avg_12h = avg(vals_12h)

    # ---- 3 · ensure no writes ---------------------------------------
    if check_has_post(results):
        print("Failure: found POST/PUT but this is a read-only task.")
        return False

    # ---- 4 · parse agent output robustly -----------------------------
    raw = results.result
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        agent_avg_6h, agent_avg_12h = raw
    else:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            try:
                parsed = json.loads(raw)
            except Exception:
                print("Failed to parse agent result:", raw)
                return False
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
            print("Unexpected result shape:", parsed)
            return False
        agent_avg_6h, agent_avg_12h = parsed

    # ---- 5 · compare with tolerance ---------------------------------
    def normalize_agent(v):
        if v is None:
            return None
        if v == -1:
            return None
        if isinstance(v, str) and v.strip() == "-1":
            return None
        return v

    agent_avg_6h = normalize_agent(agent_avg_6h)
    agent_avg_12h = normalize_agent(agent_avg_12h)

    def close(a, b, tol=0.1):
        if a is None and b is None:
            return True
        if (a is None) ^ (b is None):
            return False
        try:
            return abs(float(a) - float(b)) <= tol
        except Exception:
            return False

    return close(agent_avg_6h, gold_avg_6h) and close(agent_avg_12h, gold_avg_12h)


def task4(case_data, results, fhir_api_base):
    """
    Task 4 – Urinary-catheter dwell check (minimal-check + LASTUPDATED cut-off)

    Core logic
    ----------
    • Catheter insertions are Procedure **NUR1373**.
    • Consider only resources whose `meta.lastUpdated` is **on or before
      2025-01-01 00:00 UTC** (baseline cut-off).
    • If the latest such insertion is > 48 h old at the task’s NOW
      (2023-11-11 09:00 UTC) **and** there is no active removal order that also
      predates the cut-off, the agent must post ≥ 1 ServiceRequest (status
      active, intent order) that contains **NUR1373** in its code/text.
    • Otherwise, the agent should post nothing.

    Only essential checks:
      – subject.reference matches
      – resourceType = ServiceRequest
      – status = active, intent = order
      – “NUR1373” appears in the code/text
    """
    NOW = extract_now_from_context(case_data)
    FORTY_EIGHT_H = timedelta(hours=48)
    PATIENT_REF = {"reference": f"Patient/{case_data['eval_MRN']}"}
    PROC_CODE = "NUR1373"

    # ---------- 1 · latest catheter insertion before cut-off ----------
    proc_url = (
        f"{fhir_api_base}Procedure?"
        f"patient={case_data['eval_MRN']}&code={PROC_CODE}&_count=5000&_format=json"
    )
    insert_entries = [
        e
        for e in json.loads(send_get_request(proc_url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    last_insert = None
    for e in insert_entries:
        dt = (
            e["resource"].get("performedDateTime")
            or e["resource"].get("performedPeriod", {}).get("end")
            or e["resource"].get("performedPeriod", {}).get("start")
        )
        if dt:
            ts = datetime.fromisoformat(dt)
            if last_insert is None or ts > last_insert:
                last_insert = ts

    # ---------- 2 · existing active removal orders before cut-off -----
    sr_url = (
        f"{fhir_api_base}ServiceRequest?"
        f"patient={case_data['eval_MRN']}&code={PROC_CODE}"
        f"&status=active&_count=5000&_format=json"
    )
    active_removals = [
        e
        for e in json.loads(send_get_request(sr_url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    removal_required = (
        last_insert is not None
        and NOW - last_insert > FORTY_EIGHT_H
        and not active_removals
    )

    # ---------- 3 · evaluate agent posts ------------------------------
    posts = extract_posts(results)

    if not removal_required:
        return not posts  # should be silent

    for _, payload in posts:
        if (
            payload.get("resourceType") == "ServiceRequest"
            and payload.get("status", "").lower() == "active"
            and payload.get("intent", "").lower() == "order"
            and payload.get("subject") == PATIENT_REF
            and PROC_CODE.lower() in str(payload.get("code", "")).lower()
        ):
            return True

    return False


def task5(case_data, results, fhir_api_base):
    """
    Task 5 – Renal-mass follow-up (minimal-check + LASTUPDATED cut-off)

    Core rule (unchanged)
    ---------------------
    • Confirm malignant neoplasm of left kidney (ICD-10 C64.2)
      *recorded on or before the cut-off*.
    • If present and the most-recent CT Abd/Pelvis (IMGCT0491, IMGIL0001, or
      CPT 74177) *on or before the cut-off* is > 3 months old relative to the
      task’s NOW, order:
        – CT Abd/Pelvis w/ IV contrast (CPT 74177)
        – IR referral (CPT CON417)
    • Otherwise, no orders required.

    Essential checks only:
      – subject.reference correct
      – resourceType ∈ {ServiceRequest, MedicationRequest}
      – status = active, intent = order
      – required CPTs present (74177 and CON417)
    """
    NOW = extract_now_from_context(case_data)  # 2022-07-20 07:00 UTC
    THREE_MONTHS = timedelta(days=90)
    PATIENT_REF = {"reference": f"Patient/{case_data['eval_MRN']}"}

    DX_CODE = "C64.2"
    CT_CODE = "74177"
    IR_CODE = "CON417"
    CT_PRIOR = ["IMGCT0491", "IMGIL0001", CT_CODE]

    # ------------- 1 · verify diagnosis before cut-off -------------
    dx_url = (
        f"{fhir_api_base}Condition?"
        f"patient={case_data['eval_MRN']}&code={DX_CODE}&_count=5000&_format=json"
    )
    dx_entries = [
        e
        for e in json.loads(send_get_request(dx_url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]
    if not dx_entries:
        return not check_has_post(results)  # no dx → expect silence

    # ------------- 2 · most-recent CT before cut-off --------------
    proc_url = (
        f"{fhir_api_base}Procedure?"
        f"patient={case_data['eval_MRN']}&code={','.join(CT_PRIOR)}"
        f"&_count=5000&_format=json"
    )
    ct_entries = [
        e
        for e in json.loads(send_get_request(proc_url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    last_ct = None
    for e in ct_entries:
        dt = (
            e["resource"].get("performedDateTime")
            or e["resource"].get("performedPeriod", {}).get("end")
            or e["resource"].get("performedPeriod", {}).get("start")
        )
        if dt:
            ts = datetime.fromisoformat(dt)
            if (last_ct is None) or (ts > last_ct):
                last_ct = ts

    needs_orders = (last_ct is None) or (NOW - last_ct > THREE_MONTHS)

    # ------------- 3 · evaluate agent actions ----------------------
    posts = extract_posts(results)

    if not needs_orders:
        return not posts

    found_ct = False
    found_ir = False
    for _, pl in posts:
        if pl.get("resourceType") not in {"ServiceRequest", "MedicationRequest"}:
            continue
        if pl.get("status", "").lower() != "active":
            continue
        if pl.get("intent", "").lower() != "order":
            continue
        if pl.get("subject") != PATIENT_REF:
            continue

        if CT_CODE in str(pl.get("code", {})):
            found_ct = True
        if IR_CODE in str(pl.get("code", {})):
            found_ir = True

    return found_ct and found_ir


def task6(case_data, results, fhir_api_base):
    """
    Task 6 – Thyroid-function protocol (minimal-check + LASTUPDATED cut-off)

    Core rules (unchanged)
    ----------------------
    • Collect TSH (“TSH”) and free T4 (“FT4”) Observations *up to the cut-off*.
      ─ Branch A – ≥2 TSH > 10 µIU/mL, ≥30 d apart  → levothyroxine 25 µg PO
        + repeat TSH + FT4 orders.
      ─ Branch B – latest TSH 4.5–10 & latest FT4 0.8–1.8 ng/dL → repeat labs
        only.
      ─ Otherwise → no orders.

    Essential checks only:
      • subject.reference correct
      • resourceType / status / intent correct
      • “levothyroxine” + “25” in med text
      • “TSH” or “FT4” in lab order text/code
    """
    PATIENT_REF = {"reference": f"Patient/{case_data['eval_MRN']}"}

    # ---------- helpers ----------
    def fetch_obs(code):
        url = (
            f"{fhir_api_base}Observation?"
            f"patient={case_data['eval_MRN']}&code={code}&_count=5000&_format=json"
        )
        return [
            e
            for e in json.loads(send_get_request(url)["data"]).get("entry", [])
            if before_cutoff(e["resource"])
        ]

    def val_dt(entry):
        v = entry["resource"]["valueQuantity"]["value"]
        d = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
        return v, d

    # ---------- 1 · collect pre-cut-off labs ----------
    tsh_vals = [val_dt(e) for e in fetch_obs("TSH")]
    ft4_vals = [val_dt(e) for e in fetch_obs("FT4")]

    last_tsh_val, last_tsh_dt = (
        max(tsh_vals, key=lambda x: x[1]) if tsh_vals else (None, None)
    )
    last_ft4_val, last_ft4_dt = (
        max(ft4_vals, key=lambda x: x[1]) if ft4_vals else (None, None)
    )

    # Branch A: two high TSH values ≥30 d apart
    high_dates = [d for v, d in tsh_vals if v > 10]
    high_dates.sort()
    branch_A = any(
        (high_dates[j] - high_dates[i]).days >= 30
        for i in range(len(high_dates))
        for j in range(i + 1, len(high_dates))
    )

    # Branch B: sub-clinical hypothyroid pattern
    branch_B = (
        last_tsh_val is not None
        and 4.5 <= last_tsh_val <= 10
        and last_ft4_val is not None
        and 0.8 <= last_ft4_val <= 1.8
    )

    # ---------- 2 · evaluate agent actions ----------
    posts = extract_posts(results)
    meds = [p for p in posts if p[1].get("resourceType") == "MedicationRequest"]
    labs = [p for p in posts if p[1].get("resourceType") == "ServiceRequest"]

    def ok_med(p):
        txt = p[1]["medicationCodeableConcept"].get("text", "").lower()
        return (
            p[1].get("status", "").lower() == "active"
            and p[1].get("intent", "").lower() == "order"
            and p[1].get("subject") == PATIENT_REF
            and "levothyroxine" in txt
            and "25" in txt
        )

    def ok_lab(p):
        txt = (
            str(p[1].get("code", {})).lower()
            + p[1].get("code", {}).get("text", "").lower()
        )
        return (
            p[1].get("status", "").lower() == "active"
            and p[1].get("intent", "").lower() == "order"
            and p[1].get("subject") == PATIENT_REF
            and ("tsh" in txt or "ft4" in txt)
        )

    # ---------- 3 · branch validation ----------
    if branch_A:
        return any(ok_med(p) for p in meds) and sum(ok_lab(p) for p in labs) >= 2

    if branch_B:
        return (not meds) and sum(ok_lab(p) for p in labs) >= 2

    # No branch triggered → expect no posts
    return not posts


def task7(case_data, results, fhir_api_base):
    """
    Task 7 – Prolonged-QT management (minimal-check + LASTUPDATED cut-off)

    Core rule (unchanged)
    ---------------------
    • Latest QTc (code “QTCINTERVAL”) **before the cut-off** decides:
        – QTc ≤ 500 ms → agent must post *nothing*.
        – QTc  > 500 ms → agent must:
            ① order at least one 12-lead ECG  (SNOMED 445118002 or “ECG” text),
            ② discontinue ≥ 1 QT-prolonging drug (ondansetron, prochlorperazine,
               common antipsychotics).

    Baseline records considered ONLY if  meta.lastUpdated ≤ CUTOFF.
    """
    # ---------------- constant tables ------------------
    QT_THRESHOLD = 500
    ECG_CODE = "445118002"
    QT_MED_WORDS = [
        "ondansetron",
        "prochlorperazine",
        "haloperidol",
        "quetiapine",
        "olanzapine",
        "risperidone",
        "ziprasidone",
        "clozapine",
        "chlorpromazine",
    ]
    NON_ACTIVE = {"stopped", "cancelled", "completed", "entered-in-error"}
    PATIENT_REF = {"reference": f"Patient/{case_data['eval_MRN']}"}

    # ---------------- 1 · QTc before cut-off -------------------
    url_qt = (
        f"{fhir_api_base}Observation?"
        f"patient={case_data['eval_MRN']}&code=QTCINTERVAL"
        f"&_count=5000&_format=json"
    )
    qt_entries = [
        e
        for e in json.loads(send_get_request(url_qt)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    if not qt_entries:  # no baseline data → agent silent
        return not check_has_post(results)

    # pick the newest pre-cut-off QTc
    qt_latest = max(
        qt_entries, key=lambda e: e["resource"]["meta"].get("lastUpdated", "")
    )["resource"]
    qt_val = qt_latest["valueQuantity"]["value"]
    prolonged = qt_val > QT_THRESHOLD

    # ---------------- 3 · evaluate agent posts -----------------
    posts = extract_posts(results)

    if not prolonged:  # QTc normal → expect silence
        return not posts

    # QTc prolonged → need ≥1 ECG order & ≥1 med discontinue
    found_ecg, found_stop = False, False

    for _, pl in posts:
        rtype = pl.get("resourceType")
        if rtype not in {"ServiceRequest", "MedicationRequest"}:
            return False
        if pl.get("subject") != PATIENT_REF:
            return False

        blob = str(pl).lower()

        # ECG order?
        if rtype == "ServiceRequest" and pl.get("status", "").lower() == "active":
            if ECG_CODE in blob or "ecg" in blob:
                found_ecg = True

        # Med discontinue?
        if rtype == "MedicationRequest" and pl.get("status", "").lower() in NON_ACTIVE:
            if any(w in blob for w in QT_MED_WORDS):
                found_stop = True

    return found_ecg and found_stop


def task8(case_data, results, fhir_api_base):
    """
    Task 8 – Naloxone “rescue” coverage  (minimal-check + LASTUPDATED cutoff)

    Core rule (unchanged)
    ---------------------
    • If, **as of 2025-01-01 00:00 UTC**, the patient has ≥ 1 active opioid
      analgesic order (hydromorphone, oxycodone, fentanyl, hydrocodone, morphine)
      and **no** active naloxone / Narcan order, the agent must add exactly one
      naloxone rescue order.
    • Otherwise, the agent should post nothing.

    We now use `meta.lastUpdated` to decide whether a MedicationRequest existed
    before the cut-off.
    """
    PATIENT_REF = {"reference": f"Patient/{case_data['eval_MRN']}"}
    OPIOID_WORDS = ["hydromorphone", "oxycodone", "fentanyl", "hydrocodone", "morphine"]

    def med_text(entry):
        return entry["resource"]["medicationCodeableConcept"].get("text", "").lower()

    # ---------------- 1 · active MedicationRequests ----------------
    url = (
        f"{fhir_api_base}MedicationRequest?"
        f"patient={case_data['eval_MRN']}&status=active&_count=5000&_format=json"
    )
    entries = json.loads(send_get_request(url)["data"]).get("entry", [])

    def before_cutoff_8(entry):
        ts = entry["resource"]["meta"].get("lastUpdated")
        return ts is None or datetime.fromisoformat(ts) <= CUTOFF

    # keep only pre-cut-off records
    baseline = [e for e in entries if before_cutoff_8(e)]

    active_opioids = [
        e for e in baseline if any(w in med_text(e) for w in OPIOID_WORDS)
    ]
    active_nalox = [
        e for e in baseline if "naloxone" in med_text(e) or "narcan" in med_text(e)
    ]

    needs_naloxone = bool(active_opioids) and not active_nalox

    # ---------------- 2 · agent’s new POSTs ----------------
    posts = extract_posts(results)

    if not needs_naloxone:
        return not posts  # agent should do nothing

    # Must add exactly one naloxone order
    if len(posts) != 1:
        return False

    _, pl = posts[0]

    if pl.get("resourceType") not in {"ServiceRequest", "MedicationRequest"}:
        return False
    if pl.get("status", "").lower() != "active":
        return False
    if pl.get("intent", "").lower() != "order":
        return False
    if pl.get("subject") != PATIENT_REF:
        return False

    # text / coding must clearly reference naloxone
    blob = str(pl).lower()
    return ("naloxone" in blob) or ("narcan" in blob)


def task9(case_data, results, fhir_api_base):
    """
    Task 9 – Influenza-vaccine recall (minimal-check + LASTUPDATED cut-off)

    • Consider only prior influenza vaccinations (CPT 90686) whose
      meta.lastUpdated is on or before 2025-01-01 00:00 UTC.
    • If the newest such shot is ≤ 365 days old at NOW
      → agent must post nothing.
    • Otherwise → agent must post exactly one new flu-shot order
      (ServiceRequest or MedicationRequest) identifiable by CPT 90686
      or flu-keyword text.
    """
    NOW = extract_now_from_context(case_data)  # 2024-01-09 00:00 UTC
    ONE_YEAR = timedelta(days=365)
    CPT_FLU = "90686"
    PATIENT_REF = f"Patient/{case_data['eval_MRN']}"

    # ---------- 1 · prior flu shots before cut-off ----------
    url = (
        f"{fhir_api_base}Procedure?"
        f"patient={case_data['eval_MRN']}&code={CPT_FLU}&_count=5000&_format=json"
    )
    proc_entries = [
        e
        for e in json.loads(send_get_request(url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    last_vax_dt = None
    for e in proc_entries:
        performed = (
            e["resource"].get("performedDateTime")
            or e["resource"].get("performedPeriod", {}).get("end")
            or e["resource"].get("performedPeriod", {}).get("start")
        )
        if performed:
            ts = datetime.fromisoformat(performed)
            if last_vax_dt is None or ts > last_vax_dt:
                last_vax_dt = ts

    needs_vax = (last_vax_dt is None) or ((NOW - last_vax_dt) > ONE_YEAR)

    # ---------- 2 · examine agent’s POSTs ----------
    posts = extract_posts(results)

    if not needs_vax:
        return not posts  # agent should be silent

    if len(posts) != 1:
        return False

    _, payload = posts[0]
    rtype = payload.get("resourceType")
    if rtype not in {"ServiceRequest", "MedicationRequest"}:
        return False
    if payload.get("status", "").lower() != "active":
        return False
    if payload.get("intent", "").lower() != "order":
        return False
    if payload.get("subject") != {"reference": PATIENT_REF}:
        return False

    # recognisably a flu shot
    flu_ok = False
    if "code" in payload and "coding" in payload["code"]:
        flu_ok = payload["code"]["coding"][0].get("code") == CPT_FLU
    if not flu_ok:
        flu_ok = "flu" in str(payload).lower() or "influenza" in str(payload).lower()

    return flu_ok


def task10(case_data, results, fhir_api_base):
    """
    Task 10 – COVID-19 booster (minimal-check + LASTUPDATED cut-off)

    Rules (same as before)
    ----------------------
    • Examine the latest COVID vaccination **recorded on or before
      2025-01-01 00:00 UTC**.
        – If it is ≤ 12 months old at NOW (2023-11-07 12:00 UTC) → agent must
          post **nothing**.
        – Otherwise → agent must post **exactly one** new COVID-booster order
          (ServiceRequest or MedicationRequest).
    • Essential checks only:
        – subject.reference correct
        – resourceType ∈ {ServiceRequest, MedicationRequest}
        – status = active, intent = order
        – text/code mentions “COVID”

    The cut-off is applied with `meta.lastUpdated`.
    """
    NOW = extract_now_from_context(case_data)
    PATIENT_REF = f"Patient/{case_data['eval_MRN']}"
    PROC_CODE = "COVIDVACCINE"

    # ---------------- 1 · prior COVID vaccinations (pre-cut-off) ----------------
    proc_url = (
        f"{fhir_api_base}Procedure?"
        f"patient={case_data['eval_MRN']}&code={PROC_CODE}&_count=5000&_format=json"
    )
    proc_entries = [
        e
        for e in json.loads(send_get_request(proc_url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
    ]

    # MedicationRequest records containing “COVID-19 VAC”
    med_url = (
        f"{fhir_api_base}MedicationRequest?"
        f"patient={case_data['eval_MRN']}&status=completed&_count=5000&_format=json"
    )
    med_entries = [
        e
        for e in json.loads(send_get_request(med_url)["data"]).get("entry", [])
        if before_cutoff(e["resource"])
        and "covid-19 vac"
        in e["resource"]["medicationCodeableConcept"].get("text", "").lower()
    ]

    # ---------------- 1b · extract dates ----------------
    def extract_dt(res):
        return (
            res.get("performedDateTime")
            or res.get("performedPeriod", {}).get("end")
            or res.get("performedPeriod", {}).get("start")
            or res.get("authoredOn")
        )

    dates = [
        datetime.fromisoformat(dt_str)
        for e in proc_entries + med_entries
        if (dt_str := extract_dt(e["resource"]))
    ]

    last_vax = max(dates) if dates else None
    delta_days = (NOW.date() - last_vax.date()).days if last_vax else None
    needs_booster = (last_vax is None) or (delta_days > 365)

    # ---------------- 2 · evaluate agent’s POSTs ----------------
    posts = extract_posts(results)

    if not needs_booster:
        print("doesnt need booster")
        return not posts  # agent should be silent

    if len(posts) != 1:
        print("needs booster")
        return False

    _, payload = posts[0]
    if payload.get("resourceType") not in {"ServiceRequest", "MedicationRequest"}:
        return False
    if payload.get("status", "").lower() != "active":
        return False
    if payload.get("intent", "").lower() != "order":
        return False
    if payload.get("subject") != {"reference": PATIENT_REF}:
        return False

    # text/code must clearly reference a COVID vaccine / booster
    searchable = (
        str(payload.get("code", {})) + str(payload.get("medicationCodeableConcept", {}))
    ).lower()
    return "covid" in searchable
