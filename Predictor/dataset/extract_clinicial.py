import pandas as pd, re, json, datetime, pathlib

XLSX = "MU_Glioma_Post/MU-Glioma-Post_ClinicalData-July2025 (1).xlsx"
SHEET = "MU Glioma Post"
PIDTP_PATH = "MU_Glioma_Post/pidtime_combo.txt"
OUT  = "MU_Glioma_Post/glioma_aligned_to_mri_tp_v7_with_genomics_and_doses.json"

def safe_int(x):
    try:
        if pd.isna(x): return None
        return int(float(str(x).strip()))
    except: return None

def safe_dose(x):
    try:
        if pd.isna(x): return None
        return str(x)
    except: return None

def safe_float(x):
    try:
        if pd.isna(x): return None
        return float(str(x).split(' ')[0])
    except: return None

def parse_grade(x):
    if pd.isna(x): return None
    s = str(x)
    m = re.search(r'([1-4])', s)
    return int(m.group(1)) if m else None

def is_yes(x):
    if pd.isna(x): return False
    s = str(x).strip().lower()
    return s in {"1","yes","y","true","t"}

def therapy_exists(flag_val=None, name_or_type=None):
    # 处理 flag_val
    flag_check = is_yes(flag_val) if flag_val is not None else False
    
    # 处理 name_or_type，排除 NA/N/A/na/空值
    if name_or_type is not None:
        name_str = str(name_or_type).strip().upper()
        # 排除常见的"缺失值"表示
        invalid_values = {'NA', 'N/A', 'N.A.', 'NONE', 'NULL', '', 'NAN'}
        name_check = name_str not in invalid_values and bool(name_or_type)
    else:
        name_check = False
    
    return flag_check or name_check

def clip_interval(start_day, end_day, lo_excl, hi_inc):
    s = -float("inf") if start_day is None else start_day
    e =  float("inf") if end_day   is None else end_day
    cs = max(s, lo_excl+1)
    ce = min(e, hi_inc)
    return (int(cs), int(ce)) if cs <= ce else None

def place_interval_across_windows(timeline, mri, cat_key, item_start, item_end, item_payload):
    for idx, (_, mi) in enumerate(mri):
        lo = mri[idx-1][1] if idx-1 >= 0 else 0
        hi = mi
        clip = clip_interval(item_start, item_end, lo, hi)
        if clip:
            s,e = clip
            out = {k:v for k,v in item_payload.items() if v is not None}
            out["interval_start_day"] = s
            out["interval_end_day"]   = e
            timeline[idx]["actions"].setdefault(cat_key, []).append(out)

def place_single_tp(timeline, tp_index, cat_key, item_payload, reason):
    out = {k:v for k,v in item_payload.items() if v is not None}
    out["duration_unknown"] = True
    out["assigned_reason"]  = reason
    timeline[tp_index]["actions"].setdefault(cat_key, []).append(out)

def tri_numeric(x, default=2):
    if pd.isna(x): return default
    s = str(x).strip().lower()
    if s in {"0","1","2"}: return int(s)
    if s in {"no","absent","negative"}: return 0
    if s in {"yes","present","positive","methylated","mutated","amplified","gain","deletion"}: return 1
    if s in {"unknown","indeterminate","na","unable to assess"}: return 2
    try:
        f = float(s)
        if f in (0.0,1.0,2.0): return int(f)
    except: pass
    return default

# load clinical
df = pd.read_excel(XLSX, sheet_name=SHEET)

# locate MRI columns
MRI_COLS = []
pat = re.compile(r'Number of Days from Diagnosis to \d+(st|nd|rd|th) MRI \(Timepoint_(\d+)\)', re.I)
for c in df.columns:
    m = pat.search(c)
    if m:
        MRI_COLS.append((c, int(m.group(2))))
MRI_COLS.sort(key=lambda x: x[1])

# load allowed (pid, tp) from MRI dataset (你的白名单)
pid_tp_set = {}
with open(PIDTP_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if "\\" in line: pid, tp = line.split("\\", 1)
        elif "/" in line: pid, tp = line.split("/", 1)
        else: continue
        pid = pid.strip()
        m = re.search(r'Timepoint_(\d+)', tp)
        if not m: continue
        tpnum = int(m.group(1))
        pid_tp_set.setdefault(pid, set()).add(tpnum)

patients = {}
imputed_stats = {"total_imputed": 0, "leading": 0, "trailing": 0, "interior": 0, "skipped_tp_not_in_mri": 0}

for _, row in df.iterrows():
    pid_raw = row.get("Patient_ID")
    if isinstance(pid_raw, (int,float)) and not pd.isna(pid_raw):
        pid = f"PatientID_{int(pid_raw):04d}"
    elif isinstance(pid_raw, str):
        pid = pid_raw.strip()
    else:
        pid = "UNKNOWN_ID"

    allowed_tp = pid_tp_set.get(pid, set())

    # ---------- (1) 修补时间轴（先修再用），min clamp = 0 ----------
    tp_to_day = {}
    tp_present_clinical = set()
    for col, tpnum in MRI_COLS:
        val = row.get(col)
        tp_present_clinical.add(tpnum)
        v = safe_int(val)
        tp_to_day[tpnum] = (max(v, 0) if v is not None else None)

    # 仅保留 MRI 白名单里的 TP
    tp_candidates = sorted([tp for tp in tp_present_clinical if tp in allowed_tp])
    imputed_stats["skipped_tp_not_in_mri"] += len([tp for tp in tp_present_clinical if tp not in allowed_tp])

    if not tp_candidates:
        patients[pid] = {"context_static": {}, "timeline": []}
        continue

    days = {tp: tp_to_day.get(tp, None) for tp in tp_candidates}
    for idx, tp in enumerate(tp_candidates):
        if days[tp] is not None:
            days[tp] = max(days[tp], 0)  # 再次兜底
            continue
        prev_tp = next((tp_candidates[j] for j in range(idx-1, -1, -1) if days[tp_candidates[j]] is not None), None)
        next_tp = next((tp_candidates[j] for j in range(idx+1, len(tp_candidates)) if days[tp_candidates[j]] is not None), None)
        if prev_tp is not None and next_tp is not None:
            pv, nv = days[prev_tp], days[next_tp]
            imputed = (pv + nv) // 2
            days[tp] = max(imputed, 0)
            imputed_stats["interior"] += 1; imputed_stats["total_imputed"] += 1
        elif prev_tp is None and next_tp is not None:
            imputed = days[next_tp] - 100
            days[tp] = max(imputed, 0)
            imputed_stats["leading"] += 1; imputed_stats["total_imputed"] += 1
        elif prev_tp is not None and next_tp is None:
            imputed = days[prev_tp] + 100
            days[tp] = max(imputed, 0)
            imputed_stats["trailing"] += 1; imputed_stats["total_imputed"] += 1
        else:
            days[tp] = 0
            imputed_stats["leading"] += 1; imputed_stats["total_imputed"] += 1

    mri = sorted([(tp, days[tp]) for tp in tp_candidates if days[tp] is not None],
                 key=lambda x: (x[1], x[0]))

    # ---------- (2) context_static + genomics ----------
    genomics = {
        "idh1": tri_numeric(row.get('IDH1 mutation'), 2),
        "idh2": tri_numeric(row.get('IDH2 mutation'), 2),
        "atrx": tri_numeric(row.get('ATRX mutation'), 2),
        "mgmt_methylation": tri_numeric(row.get('MGMT methylation'), 2),
        "braf_v600e": tri_numeric(row.get('BRAF V600E mutation'), 2),
        "tert_promoter": tri_numeric(row.get('TERT promoter mutation'), 2),
        "chr7_gain_chr10_loss": tri_numeric(row.get('Chromosome 7 gain and Chromosome 10 loss'), 2),
        "h3_3a": tri_numeric(row.get('H3-3A mutation'), 2),
        "egfr_amp": tri_numeric(row.get('EGFR amplification'), 2),
        "pten": tri_numeric(row.get('PTEN mutation'), 2),
        "cdkn2ab_deletion": tri_numeric(row.get('CDKN2A/B deletion'), 2),
        "tp53_alteration": tri_numeric(row.get('TP53 alteration'), 2),
        "codeletion_1p19q_detail": (str(row.get('1p/19q')).strip() if not pd.isna(row.get('1p/19q')) else None),
        "other_mutations_text": (str(row.get('Other mutations/alterations')).strip() if 'Other mutations/alterations' in df.columns and not pd.isna(row.get('Other mutations/alterations')) else None)
    }

    static = {
        "sex_at_birth": (str(row.get('Sex at Birth')).strip().lower()
                         if not pd.isna(row.get('Sex at Birth')) else None),
        "race": (str(row.get('Race')).strip().lower()
                 if not pd.isna(row.get('Race')) else None),
        "age_at_diagnosis_years": safe_float(row.get('Age at diagnosis')),
        "primary_diagnosis": (str(row.get('Primary Diagnosis')).strip().lower()
                              if not pd.isna(row.get('Primary Diagnosis')) else None),
        "who_grade": parse_grade(row.get('Grade of Primary Brain Tumor')),
        "genomics": genomics
    }

    if not mri:
        patients[pid] = {"context_static": static, "timeline": []}
        continue

    first_idx, last_idx = 0, len(mri)-1
    L = max(d for _,d in mri)

    # ---------- (3) timeline skeleton ----------
    timeline = []
    for (tpn, day) in mri:
        timeline.append({
            "tp_id": f"TP{tpn}",
            "mri_day": int(day),
            "state": { "days_since_dx": int(day) },
            "actions": {}   # 必出现，可为空字典
        })

    # ---------- (4) therapies enriched ----------
    # Radiation
    rad_exists = therapy_exists(flag_val=row.get('Radiation Therapy'))
    rs = safe_int(row.get('Number of days from Diagnosis to Radiation Therapy Start date'))
    re_ = safe_int(row.get('Number of days from Diagnosis to Radiation Therapy end date'))
    if rad_exists:
        payload = {
                "dose_gy": safe_float(row.get('Dose')),
                "fractions": safe_int(row.get('Number of Fractions')),
                "start_day": rs, "end_day": re_,
            }
        
        if rs is None and re_ is None:
            place_interval_across_windows(timeline, mri, "radiation", None, None, payload)
        elif (rs is not None and rs < 0) or (re_ is not None and re_ < 0):
            place_single_tp(timeline, first_idx, "radiation", payload, "negative_time")
        elif (rs is not None and rs > L) or (re_ is not None and re_ > L):
            place_single_tp(timeline, last_idx, "radiation", payload, "beyond_last_tp")
        else:
            place_interval_across_windows(timeline, mri, "radiation", rs, re_, payload)

    # Initial Chemo
    chemo_name = (str(row.get('Name of Initial Chemo Therapy')).strip()
                  if not pd.isna(row.get('Name of Initial Chemo Therapy')) else None)
    chemo_exists = therapy_exists(flag_val=row.get('Initial Chemo Therapy'), name_or_type=chemo_name)
    cs = safe_int(row.get(' Number of days from Diagnosis to Initial Chemo Therapy Start date'))
    ce = safe_int(row.get(' Number of days from Diagnosis to Initial Chemo Therapy end date'))
    payload_chemo = {
        "agent": chemo_name, "start_day": cs, "end_day": ce,
        "cycle_length_days": safe_int(row.get('Cycle length of Initial Chemotherapy (q days)')) if 'Cycle length of Initial Chemotherapy (q days)' in df.columns else None,
        "num_cycles": safe_int(row.get('Number of Cycles of Initial Chemotherapy')) if 'Number of Cycles of Initial Chemotherapy' in df.columns else None,
        "dose": safe_float(row.get('Dose of Initial Chemotherapy')) if 'Dose of Initial Chemotherapy' in df.columns else None,
    }
    if chemo_exists:
        if cs is None and ce is None:
            place_interval_across_windows(timeline, mri, "chemotherapy", None, None, payload_chemo)
        elif (cs is not None and cs < 0) or (ce is not None and ce < 0):
            place_single_tp(timeline, first_idx, "chemotherapy", payload_chemo, "negative_time")
        elif (cs is not None and cs > L) or (ce is not None and ce > L):
            place_single_tp(timeline, last_idx, "chemotherapy", payload_chemo, "beyond_last_tp")
        else:
            place_interval_across_windows(timeline, mri, "chemotherapy", cs, ce, payload_chemo)

    # Immunotherapy
    im_exists = therapy_exists(flag_val=None,  name_or_type=row.get('Immuno therapy'))
    ims = safe_int(row.get('Number of Days from Diagnosis to Start Immunotherapy '))
    ime = safe_int(row.get('Number of Days from Diagnosis to Complete Immunotherapy '))
    im_drug = row.get('Immuno therapy')

    payload_im = {
        "agent": im_drug, "start_day": ims, "end_day": ime,
        "cycle_length_days": safe_int(row.get('Cycle length of Immunotherapy (q days)')) if 'Cycle length of Immunotherapy (q days)' in df.columns else None,
        "num_cycles": safe_int(row.get('Number of Cycles of Immunotherapy')) if 'Number of Cycles of Immunotherapy' in df.columns else None,
        "dose": safe_float(row.get('Dose of Immunotherapy')) if 'Dose of Immunotherapy' in df.columns else None,
    }
    if im_exists:
        if ims is None and ime is None:
            place_interval_across_windows(timeline, mri, "immunotherapy", None, None, payload_im)
        elif (ims is not None and ims < 0) or (ime is not None and ime < 0):
            place_single_tp(timeline, first_idx, "immunotherapy", payload_im, "negative_time")
        elif (ims is not None and ims > L) or (ime is not None and ime > L):
            place_single_tp(timeline, last_idx, "immunotherapy", payload_im, "beyond_last_tp")
        else:
            place_interval_across_windows(timeline, mri, "immunotherapy", ims, ime, payload_im)

    # Additional therapy 1
    a1_exists = therapy_exists(name_or_type=row.get('Additional Therapy'))
    a1_drug_name = row.get('Additional Therapy')
    a1s = safe_int(row.get('Number of Days from Diagnosis to Starting Additional Therapy '))
    a1e = safe_int(row.get('Number of Days from Diagnosis to Complete Additional Therapy '))
    payload_a1 = {
        "agent":a1_drug_name, "start_day": a1s, "end_day": a1e,
        "cycle_length_days": safe_int(row.get('Cycle length of Additional Therapy (q days)')),
        "num_cycles": safe_int(row.get('Number of Cycles of Additional Therapy')),
        "dose": safe_float(row.get('Dose of Additional Therapy')) if 'Dose of Additional Therapy' in df.columns else None,
    }
    if a1_exists:
        if a1s is None and a1e is None:
            place_interval_across_windows(timeline, mri, "additional_1", None, None, payload_a1)
        elif (a1s is not None and a1s < 0) or (a1e is not None and a1e < 0):
            place_single_tp(timeline, first_idx, "additional_1", payload_a1, "negative_time")
        elif (a1s is not None and a1s > L) or (a1e is not None and a1e > L):
            place_single_tp(timeline, last_idx, "additional_1", payload_a1, "beyond_last_tp")
        else:
            place_interval_across_windows(timeline, mri, "additional_1", a1s, a1e, payload_a1)

    # Additional therapy 2
    a2_exists = therapy_exists(name_or_type=row.get('2nd_Additional Therapy'))
    a2_drug_name = row.get('2nd_Additional Therapy')
    a2s = safe_int(row.get('Number of Days from Diagnosis to Starting 2nd_Additional Therapy '))
    a2e = safe_int(row.get('Number of Days from Dagnosis to Complete 2nd_Additional Therapy '))
    payload_a2 = {
        "agent": a2_drug_name, "start_day": a2s, "end_day": a2e,
        "cycle_length_days": safe_int(row.get('Cycle length of 2nd_Additional Therapy (q days)')),
        "num_cycles": safe_int(row.get('Number of Cycles of 2nd_Additional Therapy')) if 'Number of Cycles of 2nd_Additional Therapy' in df.columns else None,
        "dose": safe_float(row.get('Dose of 2nd Additional Therapy')) if 'Dose of 2nd Additional Therapy' in df.columns else None,
    }
    if a2_exists:
        if a2s is None and a2e is None:
            place_interval_across_windows(timeline, mri, "additional_2", None, None, payload_a2)
        elif (a2s is not None and a2s < 0) or (a2e is not None and a2e < 0):
            place_single_tp(timeline, first_idx, "additional_2", payload_a2, "negative_time")
        elif (a2s is not None and a2s > L) or (a2e is not None and a2e > L):
            place_single_tp(timeline, last_idx, "additional_2", payload_a2, "beyond_last_tp")
        else:
            place_interval_across_windows(timeline, mri, "additional_2", a2s, a2e, payload_a2)

    # Other typed therapy
    ot_type = (str(row.get('Other Types of Therapy (LITT, more chemo, proton therapy)')).strip().lower()
               if not pd.isna(row.get('Other Types of Therapy (LITT, more chemo, proton therapy)')) else None)
    ot_exists = therapy_exists(name_or_type=ot_type)
    ots = safe_int(row.get('Number of Days from Diagnosis to Start Other Additional Therapy '))
    ote = safe_int(row.get('Number of Days from Diagnosis to Complete Other Additional Therapy '))
    if ot_exists:
        payload_ot = {"agent": ot_type or "other", "start_day": ots, "end_day": ote}
        if ots is None and ote is None:
            place_interval_across_windows(timeline, mri, "other", None, None, payload_ot)
        elif (ots is not None and ots < 0) or (ote is not None and ote < 0):
            place_single_tp(timeline, first_idx, "other", payload_ot, "negative_time")
        elif (ots is not None and ots > L) or (ote is not None and ote > L):
            place_single_tp(timeline, last_idx, "other", payload_ot, "beyond_last_tp")
        else:
            place_interval_across_windows(timeline, mri, "other", ots, ote, payload_ot)

    # Brachy (point)
    brachy_exists = therapy_exists(name_or_type=row.get('Brachy therapy'))
    brachy_drug_name = row.get('Brachy therapy')
    bd = safe_int(row.get('Number of Days from Diagnosis to the day of Insertion of Brachytherapy '))
    if brachy_exists:
        if bd is None:
            place_single_tp(timeline, first_idx, "brachy", {"agent": brachy_drug_name, "day_of_insertion": None}, "no_time_point_event")
        elif bd < 0:
            place_single_tp(timeline, first_idx, "brachy", {"agent": brachy_drug_name, "day_of_insertion": bd}, "negative_time_point_event")
        elif bd > L:
            place_single_tp(timeline, last_idx, "brachy", {"agent": brachy_drug_name, "day_of_insertion": bd}, "beyond_last_tp_point_event")
        else:
            for idx, (_, mi) in enumerate(mri):
                lo = mri[idx-1][1] if idx-1 >= 0 else 0
                hi = mi
                if lo < bd <= hi:
                    timeline[idx]["actions"].setdefault("brachy", []).append({"day_of_insertion": bd})
                    break

    # ---------- (5) progression & survival ----------
    prog_time = safe_int(row.get('Time to First Progression (Days)'))
    prog_type = (str(row.get('Type of 1st Progression')).strip().lower()
                 if not pd.isna(row.get('Type of 1st Progression')) else None)
    death_flag = safe_int(row.get('Overall Survival (Death)'))
    death_day  = safe_int(row.get('Number of days from Diagnosis to death (Days)'))
    special = (death_flag == 1 and death_day is not None and any(d > death_day for _,d in mri))

    for idx, (tpn, mi) in enumerate(mri):
        lo = mri[idx-1][1] if idx-1 >= 0 else 0
        occurred_up_to = 1 if (prog_time is not None and prog_time <= mi) else (1 if safe_int(row.get('Progression')) == 1 else 0)
        occurred_in_interval = 1 if (prog_time is not None and lo < prog_time <= mi) else 0
        tptype = (prog_type if occurred_up_to else "none") if prog_type else ("unknown" if occurred_up_to else "none")

        if death_flag != 1 or death_day is None:
            data_surv = {"event_indicator": 0,
                         "survival_from_tp_days": max(mri[-1][1] - mi, 0),
                         "censoring_rule": "no_death_last_mri"}
        else:
            if special:
                Dstar = mri[-1][1] + 1
                data_surv = {"event_indicator": 1,
                             "survival_from_tp_days": max(Dstar - mi, 0),
                             "censoring_rule": "death_shifted_to_L_plus_1"}
            else:
                data_surv = {"event_indicator": 1,
                             "survival_from_tp_days": max(death_day - mi, 0),
                             "censoring_rule": "death_known"}

        timeline[idx].setdefault("state", {})
        timeline[idx]["state"]["progression"] = {
            "occurred_up_to_tp": int(occurred_up_to),
            "occurred_in_interval": int(occurred_in_interval),
            "type": tptype,
            "first_progression_day": (prog_time if occurred_up_to else None)
        }
        timeline[idx]["survival"] = data_surv

    patients[pid] = {"context_static": static, "timeline": timeline}

# ---------- save ----------
merged = {
    "schema_version": "glioma_meta_v7_tp_repair_min0_with_genomics_doses",
    "merged_at": datetime.datetime.utcnow().isoformat()+"Z",
    "sources": {"xlsx": pathlib.Path(XLSX).name, "pid_tp_list": pathlib.Path(PIDTP_PATH).name},
    "imputation_summary": imputed_stats,
    "patients": patients
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Wrote:", OUT)
