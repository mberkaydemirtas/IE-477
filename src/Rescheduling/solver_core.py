#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
solver_core.py (OPEN-SOURCE ONLY)

- Baseline artık HeuristicBaseModel.py'den alınacak (run_heuristic).
- Bu dosya: rescheduling (urgent job + disruption) için heuristic üretir.
- Gurobi / MIP yok.

Reschedule mantığı:
- t0 = plan_start'tan şimdiye kadar geçen "working hour" zamanı
- I_done: C_old <= t0
- I_run : S_old < t0 < C_old
- I_free: S_old >= t0

"Tercih 1 / continue" mantığı:
- Running op normalde devam eder (kesilmez)
- AMA eğer running op'un machine veya station'ı unavailable ise => t0’da kes (preempt)
  ve remainder op schedule edilir.

Heuristic: GT pivot + ATC (batch scheduling at fixed t) — baseline ile aynı ruh.

✅ Bu sürümde kritik düzeltmeler:
1) Scenario JSON'daki `data` normalize edilir:
   - O_j/M_i/L_i/Pred_i key'leri int olur
   - p_im JSON-safe string/nested formatlardan (i,m) tuple dict'e çevrilir
2) Running op remainder için beta_i korunur (big station kısıtı kaybolmaz)
3) Fixed ops için machine/station None gelirse schedule/x_old fallback + en son feasible fallback yapılır
"""

import math
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Any, Dict, List, Tuple, Optional


# ==========================================================
#  TIME HELPERS (working-hours based t0, NO tzdata dependency)
# ==========================================================

def _parse_hhmm(hhmm: str):
    hh, mm = hhmm.strip().split(":")
    return int(hh), int(mm)

def _parse_utc_offset(s: str) -> timezone:
    s = (s or "+00:00").strip()
    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    hh, mm = s.split(":")
    delta = timedelta(hours=sign * int(hh), minutes=sign * int(mm))
    return timezone(delta)

def compute_t0_from_plan_start(plan_start_iso: str, plan_calendar: dict | None = None) -> float:
    """
    t0 = elapsed time since plan_start, measured in *working hours*.

    Rules:
    - 1 workday = workday_hours (default 8.0)
    - shift start local time = shift_start_local (default "08:00")
    - NO lunch break
    - default workdays = Mon-Fri (0..4)

    plan_start_iso is UTC ISO string (timezone-aware recommended).
    plan_calendar uses utc_offset (e.g. "+03:00") to avoid tzdata on Windows.
    """
    plan_calendar = plan_calendar or {}

    utc_offset = plan_calendar.get("utc_offset", "+03:00")  # TR default
    local_tz = _parse_utc_offset(utc_offset)

    shift_start_local = plan_calendar.get("shift_start_local", "08:00")
    workday_hours = float(plan_calendar.get("workday_hours", 8.0))

    workdays = plan_calendar.get("workdays", [0, 1, 2, 3, 4])
    workdays = set(int(x) for x in workdays)

    start_utc = datetime.fromisoformat(plan_start_iso)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)

    start_local = start_utc.astimezone(local_tz)
    now_local = datetime.now(timezone.utc).astimezone(local_tz)

    if now_local <= start_local:
        return 0.0

    sh, sm = _parse_hhmm(shift_start_local)

    def shift_window_for_date(d):
        s = datetime.combine(d, dtime(sh, sm), tzinfo=local_tz)
        e = s + timedelta(hours=workday_hours)
        return s, e

    total = 0.0
    d = start_local.date()
    today = now_local.date()

    while d < today:
        if d.weekday() in workdays:
            s, e = shift_window_for_date(d)

            if d == start_local.date():
                left = max(start_local, s)
            else:
                left = s

            right = e
            if right > left:
                total += (right - left).total_seconds() / 3600.0
        d = d + timedelta(days=1)

    if today.weekday() in workdays:
        s, e = shift_window_for_date(today)

        left = s
        if today == start_local.date():
            left = max(left, start_local)

        right = min(now_local, e)
        if right > left:
            total += (right - left).total_seconds() / 3600.0

    return float(total)


# ==========================================================
#  JSON-SAFE KEY HELPERS
# ==========================================================

def _tuplekey_to_str(i: int, k: int) -> str:
    return f"{int(i)},{int(k)}"

def _str_to_tuplekey(s: str):
    s = s.strip().replace("(", "").replace(")", "")
    a, b = [x.strip() for x in s.split(",")]
    return (int(a), int(b))


# ==========================================================
#  DATA NORMALIZATION (CRITICAL FOR SCENARIO JSON)
# ==========================================================

def _to_int_list(xs) -> List[int]:
    return [int(x) for x in (xs or [])]

def _dict_int_keys(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            ik = int(k)
        except Exception:
            ik = k
        out[ik] = v
    return out

def _dict_int_keys_list_int_vals(d: dict) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for k, v in (d or {}).items():
        out[int(k)] = [int(x) for x in (v or [])]
    return out

def _dict_int_keys_float_vals(d: dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k, v in (d or {}).items():
        out[int(k)] = float(v)
    return out

def _dict_int_keys_int_vals(d: dict) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in (d or {}).items():
        out[int(k)] = int(v)
    return out

def _normalize_p_im(p_im_raw: Any) -> Dict[Tuple[int, int], float]:
    """
    Supports common JSON formats:
    A) {"1,2": 3.5, "1,3": 4.0}
    B) {"1": {"2": 3.5, "3": 4.0}}
    C) {(1,2): 3.5}  (already tuple keys - Python dict from code)
    """
    if p_im_raw is None:
        return {}

    out: Dict[Tuple[int, int], float] = {}

    # already tuple keyed
    if isinstance(p_im_raw, dict):
        sample_key = next(iter(p_im_raw.keys()), None)

        # Case C: tuple keys
        if isinstance(sample_key, tuple) and len(sample_key) == 2:
            for (i, m), v in p_im_raw.items():
                out[(int(i), int(m))] = float(v)
            return out

        # Case A: "i,m" keys
        if isinstance(sample_key, str) and ("," in sample_key):
            for k, v in p_im_raw.items():
                i, m = _str_to_tuplekey(str(k))
                out[(int(i), int(m))] = float(v)
            return out

        # Case B: nested dict: {"i": {"m": p}}
        if isinstance(sample_key, (str, int)) and isinstance(p_im_raw.get(sample_key), dict):
            for i_key, inner in p_im_raw.items():
                i = int(i_key)
                for m_key, v in inner.items():
                    m = int(m_key)
                    out[(i, m)] = float(v)
            return out

        # fallback: try parse each key
        for k, v in p_im_raw.items():
            if isinstance(k, str) and "," in k:
                i, m = _str_to_tuplekey(k)
                out[(int(i), int(m))] = float(v)
    return out

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes scenario `data` safe for tuple indexing and int-key access.
    """
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    out = dict(data)

    # Lists
    out["J"] = _to_int_list(out.get("J"))
    out["I"] = _to_int_list(out.get("I"))
    out["M"] = _to_int_list(out.get("M"))
    out["L"] = _to_int_list(out.get("L"))

    # Dicts with int keys
    out["O_j"] = _dict_int_keys_list_int_vals(out.get("O_j", {}))
    out["M_i"] = _dict_int_keys_list_int_vals(out.get("M_i", {}))
    out["L_i"] = _dict_int_keys_list_int_vals(out.get("L_i", {}))
    out["Pred_i"] = _dict_int_keys_list_int_vals(out.get("Pred_i", {}))

    out["r_j"] = _dict_int_keys_float_vals(out.get("r_j", {}))
    out["d_j"] = _dict_int_keys_float_vals(out.get("d_j", {}))
    out["g_j"] = _dict_int_keys_int_vals(out.get("g_j", {}))

    # paint flag: accept p_flag_j OR p_j; if none => zeros
    pflag = out.get("p_flag_j", None)
    if pflag is None:
        pflag = out.get("p_j", {})
    out["p_flag_j"] = _dict_int_keys_int_vals(pflag or {})

    out["t_grind_j"] = _dict_int_keys_float_vals(out.get("t_grind_j", {}))
    out["t_paint_j"] = _dict_int_keys_float_vals(out.get("t_paint_j", {}))
    out["beta_i"] = _dict_int_keys_int_vals(out.get("beta_i", {}))

    # station sets
    out["L_small"] = _to_int_list(out.get("L_small", []))
    if "L_big" in out:
        out["L_big"] = _to_int_list(out.get("L_big", []))

    # p_im
    out["p_im"] = _normalize_p_im(out.get("p_im"))

    # Safety: if p_flag_j missing some jobs, fill zeros
    for j in out["J"]:
        if j not in out["p_flag_j"]:
            out["p_flag_j"][j] = 0
        if j not in out["g_j"]:
            out["g_j"][j] = 0
        if j not in out["r_j"]:
            out["r_j"][j] = 0.0
        if j not in out["d_j"]:
            # if due missing, put a big due (avoid crash)
            out["d_j"][j] = 1e9
        if j not in out["t_grind_j"]:
            out["t_grind_j"][j] = 0.0
        if j not in out["t_paint_j"]:
            out["t_paint_j"][j] = 0.0

    return out


# ==========================================================
#  OLD SOLUTION NORMALIZATION (BASELINE JSON)
# ==========================================================

def normalize_old_solution(old_solution: dict) -> dict:
    out = dict(old_solution)

    for k in ["S_old", "C_old"]:
        if k in out and isinstance(out[k], dict):
            out[k] = {int(kk): float(vv) for kk, vv in out[k].items()}

    for k in ["x_old", "y_old"]:
        if k in out and isinstance(out[k], dict):
            converted = {}
            for kk, vv in out[k].items():
                if isinstance(kk, str):
                    i, m = _str_to_tuplekey(kk)
                else:
                    i, m = kk
                converted[(int(i), int(m))] = int(vv)
            out[k] = converted

    return out


# ==========================================================
#  SET CLASSIFICATION
# ==========================================================

def classify_ops_by_t0(I, S_old, C_old, t0):
    done = [int(i) for i in I if C_old[int(i)] <= t0 + 1e-9]
    run  = [int(i) for i in I if (S_old[int(i)] < t0 - 1e-9) and (C_old[int(i)] > t0 + 1e-9)]
    free = [int(i) for i in I if S_old[int(i)] >= t0 - 1e-9]
    return done, run, free

def op_to_job_map(O_j):
    mp = {}
    for j, ops in O_j.items():
        for i in ops:
            mp[int(i)] = int(j)
    return mp


# ==========================================================
#  URGENT JOB (payload)
# ==========================================================

def add_urgent_job_from_payload(data, t0: float, urgent_payload: dict):
    uj = urgent_payload.get("urgent_job", urgent_payload)

    J = list(data["J"])
    I = list(data["I"])
    O_j = dict(data["O_j"])
    M = list(data["M"])
    L = list(data["L"])

    M_i = dict(data["M_i"])
    L_i = dict(data["L_i"])
    Pred_i = {int(k): list(v) for k, v in data["Pred_i"].items()}
    p_im = dict(data["p_im"])
    r_j = dict(data["r_j"])
    d_j = dict(data["d_j"])
    g_j = dict(data["g_j"])
    p_flag_j = dict(data.get("p_flag_j", data.get("p_j", {})) or {})
    t_grind_j = dict(data["t_grind_j"])
    t_paint_j = dict(data["t_paint_j"])
    beta_i = dict(data["beta_i"])

    job_id = int(uj["job_id"])
    if job_id in J:
        raise ValueError(f"urgent job_id already exists: {job_id}")

    r_mode = uj.get("release_time_mode", "t0")
    r_u = float(t0) if r_mode == "t0" else float(uj["release_time_hours"])

    d_mode = uj.get("due_time_mode", "t0_plus")
    if d_mode == "t0_plus":
        d_u = float(t0 + float(uj["due_time_hours"]))
    else:
        d_u = float(uj["due_time_hours"])

    ops_payload = uj.get("ops", [])
    if not ops_payload:
        raise ValueError("urgent_job.ops is empty")

    max_op = max(I) if I else 0
    new_ops = []
    for op in ops_payload:
        op_id = op.get("op_id", None)
        if op_id is None:
            max_op += 1
            op_id = max_op
        op_id = int(op_id)
        if op_id in I or op_id in new_ops:
            raise ValueError(f"urgent op_id clashes: {op_id}")
        new_ops.append(op_id)

    J2 = J + [job_id]
    I2 = I + new_ops
    O_j2 = dict(O_j)
    O_j2[job_id] = list(new_ops)

    for k, op_id in enumerate(new_ops):
        op_def = ops_payload[k]

        feasL = op_def.get("feasible_stations", L)
        L_i[op_id] = [int(l) for l in feasL]

        pt = op_def.get("processing_time_by_machine", {})
        if not pt:
            raise ValueError(f"urgent op {op_id} missing processing_time_by_machine")

        # pt keys might be strings in JSON
        pt_machines = sorted({int(mm) for mm in pt.keys()})

        feasM_user = op_def.get("feasible_machines", None)
        if feasM_user is None:
            feasM_final = pt_machines
        else:
            feasM_user = {int(mm) for mm in feasM_user}
            feasM_final = sorted(feasM_user.intersection(pt_machines))

        if not feasM_final:
            raise ValueError(
                f"urgent op {op_id}: feasible_machines ∩ processing_time_by_machine is empty."
            )

        M_i[op_id] = list(feasM_final)

        for m in M_i[op_id]:
            val = pt.get(str(m), None)
            if val is None and m in pt:
                val = pt[m]
            if val is None:
                raise ValueError(f"urgent op {op_id}: missing ptime for machine {m}")
            p_im[(op_id, int(m))] = float(val)

        beta_i[op_id] = 1 if op_def.get("beta_big_station_required", False) else 0
        Pred_i[op_id] = []

    edges = uj.get("precedence_edges", [])
    for a, b in edges:
        a = int(a); b = int(b)
        if a == b:
            raise ValueError(f"urgent precedence has self-edge: {a}->{b}")
        if b not in Pred_i:
            Pred_i[b] = []
        if a not in Pred_i[b]:
            Pred_i[b].append(a)

    r_j[job_id] = r_u
    d_j[job_id] = d_u

    post = uj.get("post_ops", {})
    g_req = bool(post.get("grinding_required", False))
    p_req = bool(post.get("painting_required", False))
    g_j[job_id] = 1 if g_req else 0
    p_flag_j[job_id] = 1 if p_req else 0
    t_grind_j[job_id] = float(post.get("t_grind_hours", 0.0)) if g_req else 0.0
    t_paint_j[job_id] = float(post.get("t_paint_hours", 0.0)) if p_req else 0.0

    out = dict(data)
    out.update({
        "J": J2, "I": I2, "O_j": O_j2,
        "M_i": M_i, "L_i": L_i,
        "Pred_i": Pred_i, "p_im": p_im,
        "r_j": r_j, "d_j": d_j,
        "g_j": g_j, "p_flag_j": p_flag_j,
        "t_grind_j": t_grind_j, "t_paint_j": t_paint_j,
        "beta_i": beta_i,
        "urgent_job_id": job_id,
        "urgent_ops": new_ops
    })
    return out


# ==========================================================
#  SPLIT REMAINDERS FOR RUNNING OPS
# ==========================================================

def add_split_remainders_for_running_ops(data, I_run, t0, old_solution):
    J, I, O_j = data["J"], data["I"], data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    beta_i = data["beta_i"]

    S_old = old_solution["S_old"]
    x_old = old_solution.get("x_old", {})

    next_op = max(I) + 1
    rem_map = {}
    op_to_job = op_to_job_map(O_j)

    I2 = list(I)
    O_j2 = {j: list(O_j[j]) for j in J}
    L_i2 = dict(L_i)
    M_i2 = dict(M_i)
    Pred_i2 = {k: list(Pred_i.get(k, [])) for k in I}
    p_im2 = dict(p_im)
    beta_i2 = dict(beta_i)

    def old_machine(i):
        for (ii, m), v in x_old.items():
            if int(ii) == int(i) and int(v) == 1:
                return int(m)
        return None

    for i in I_run:
        i = int(i)
        i_rem = next_op
        next_op += 1
        rem_map[i] = i_rem

        I2.append(i_rem)

        j = op_to_job.get(i, None)
        if j is not None:
            O_j2[j].append(i_rem)

        L_i2[i_rem] = list(L_i[i])
        M_i2[i_rem] = list(M_i[i])

        m0 = old_machine(i)
        if m0 is None:
            m0 = M_i[i][0]

        full_p = float(p_im[(i, m0)])
        elapsed = max(0.0, float(t0 - S_old[i]))
        remaining = max(0.01, full_p - elapsed)

        for m in M_i2[i_rem]:
            p_im2[(i_rem, int(m))] = float(remaining)

        # remainder starts after original (or after cut)
        Pred_i2[i_rem] = [i]

        # ✅ FIX: preserve big-station requirement if original had it
        beta_i2[i_rem] = int(beta_i2.get(i, 0))

        # successors of original should also depend on remainder (so work doesn't "skip")
        for k in list(Pred_i2.keys()):
            k = int(k)
            if k == i_rem or k == i:
                continue
            if i in Pred_i2.get(k, []):
                if i_rem not in Pred_i2[k]:
                    Pred_i2[k].append(i_rem)

    out = dict(data)
    out.update({
        "I": I2,
        "O_j": O_j2,
        "L_i": L_i2,
        "M_i": M_i2,
        "Pred_i": Pred_i2,
        "p_im": p_im2,
        "beta_i": beta_i2,
        "rem_map": rem_map
    })
    return out


# ==========================================================
#  RESCHEDULE HEURISTIC: GT + ATC (fixed ops + outages supported)
# ==========================================================

def _avg_pbar(data: Dict[str, Any]) -> float:
    I = data["I"]
    M_i = data["M_i"]
    p_im = data["p_im"]
    vals = []
    for i in I:
        for m in M_i[i]:
            vals.append(float(p_im[(int(i), int(m))]))
    return sum(vals) / max(1, len(vals))

def _job_of(data: Dict[str, Any]) -> Dict[int, int]:
    return op_to_job_map(data["O_j"])

def _ready_time(i: int, job_of: Dict[int, int], data: Dict[str, Any], C: Dict[int, float]) -> float:
    Pred_i = data["Pred_i"]
    r_j = data["r_j"]
    j = job_of[i]
    rt = float(r_j[j])
    preds = Pred_i.get(i, [])
    if preds:
        rt = max(rt, max(C[int(h)] for h in preds))
    return rt

def _atc_index(i: int, t_now: float, job_of: Dict[int, int], data: Dict[str, Any], pbar: float, k1: float, best_m_for_i: int) -> float:
    d_j = data["d_j"]
    p_im = data["p_im"]
    j = job_of[i]
    p_i = float(p_im[(int(i), int(best_m_for_i))])
    slack = max(float(d_j[j]) - p_i - t_now, 0.0)
    denom = max(k1 * pbar, 1e-9)
    return (1.0 / max(p_i, 1e-9)) * math.exp(-slack / denom)

def heuristic_reschedule_gt_atc(
    data: Dict[str, Any],
    fixed_ops: Dict[int, dict],
    start_time_floor: float,
    unavailable_machines: Optional[List[int]] = None,
    unavailable_stations: Optional[List[int]] = None,
    k1: float = 2.0,
    eps: float = 1e-9,
) -> Tuple[List[dict], dict]:
    """
    GT + ATC heuristic with:
    - fixed_ops pre-scheduled (block resources up to their finish)
    - start_time_floor: non-fixed ops won't start before this
    - unavailable resources: new assignments cannot use them
    """
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []
    badM = set(int(x) for x in unavailable_machines)
    badL = set(int(x) for x in unavailable_stations)

    J = data["J"]
    I = [int(x) for x in data["I"]]
    O_j = data["O_j"]
    M = data["M"]
    L = data["L"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    d_j = data["d_j"]
    g_j = data["g_j"]
    p_flag_j = data.get("p_flag_j", data.get("p_j", {})) or {}
    t_grind_j = data["t_grind_j"]
    t_paint_j = data["t_paint_j"]
    beta_i = data["beta_i"]
    L_small = set(int(x) for x in data["L_small"])

    job_of = _job_of(data)
    pbar = _avg_pbar(data)

    avail_m = {int(m): 0.0 for m in M}
    avail_l = {int(l): 0.0 for l in L}

    S: Dict[int, float] = {}
    C: Dict[int, float] = {}
    am: Dict[int, int] = {}
    al: Dict[int, int] = {}
    scheduled = set()

    # load fixed ops
    for op_id, row in (fixed_ops or {}).items():
        i = int(op_id)
        s = float(row["start"])
        c = float(row["finish"])
        m = row.get("machine", None)
        l = row.get("station", None)

        S[i] = s
        C[i] = c
        if m is not None:
            am[i] = int(m)
            avail_m[int(m)] = max(avail_m[int(m)], c)
        if l is not None:
            al[i] = int(l)
            avail_l[int(l)] = max(avail_l[int(l)], c)

        scheduled.add(i)

    t = float(start_time_floor)

    def is_pred_done(i: int) -> bool:
        preds = Pred_i.get(i, [])
        return all(int(h) in C for h in preds)

    while len(scheduled) < len(I):
        candidates = [i for i in I if i not in scheduled and is_pred_done(i)]
        if not candidates:
            raise RuntimeError("Heuristic deadlock: no eligible operations (precedence cycle?)")

        R = {i: _ready_time(i, job_of, data, C) for i in candidates}
        for i in candidates:
            R[i] = max(R[i], float(start_time_floor))

        if all(R[i] > t + eps for i in candidates):
            t = min(R[i] for i in candidates)

        # batch at fixed t
        while True:
            ready = [i for i in candidates if R[i] <= t + eps]
            if not ready:
                break

            best_pair: Dict[int, Tuple[int, int, float, float]] = {}  # i -> (m,l,s,c)

            for i in ready:
                bestS = float("inf")
                bestC = float("inf")
                bestML = None

                stations = [int(l) for l in L_i[i] if int(l) not in badL]
                if int(beta_i.get(i, 0)) == 1:
                    stations = [l for l in stations if l not in L_small]
                if not stations:
                    continue

                machines = [int(m) for m in M_i[i] if int(m) not in badM]
                if not machines:
                    continue

                for m in machines:
                    for l in stations:
                        s = max(t, R[i], avail_m.get(m, 0.0), avail_l.get(l, 0.0))
                        c = s + float(p_im[(int(i), int(m))])
                        if (s < bestS - 1e-12) or (abs(s - bestS) <= 1e-12 and c < bestC - 1e-12):
                            bestS, bestC = s, c
                            bestML = (m, l)

                if bestML is None:
                    continue

                m_best, l_best = bestML
                best_pair[i] = (m_best, l_best, bestS, bestC)

            startable = [i for i in ready if i in best_pair and abs(best_pair[i][2] - t) <= eps]
            if not startable:
                break

            i_star = min(startable, key=lambda ii: best_pair[ii][3])
            m_star, l_star, _, _ = best_pair[i_star]

            conflict = [i_star]
            for i in startable:
                if i == i_star:
                    continue
                mi, li, _, _ = best_pair[i]
                if mi == m_star or li == l_star:
                    conflict.append(i)

            def score(ii: int) -> float:
                m_best = best_pair[ii][0]
                return _atc_index(ii, t, job_of, data, pbar, k1, m_best)

            chosen = max(conflict, key=score)
            m, l, s, c = best_pair[chosen]

            S[chosen] = float(s)
            C[chosen] = float(c)
            am[chosen] = int(m)
            al[chosen] = int(l)
            avail_m[m] = float(c)
            avail_l[l] = float(c)
            scheduled.add(chosen)

            candidates.remove(chosen)
            if not candidates:
                break

            for i2 in candidates:
                if is_pred_done(i2):
                    R[i2] = max(_ready_time(i2, job_of, data, C), float(start_time_floor))

        # advance time
        if len(scheduled) < len(I):
            next_t = float("inf")

            for tt in avail_m.values():
                if tt > t + eps:
                    next_t = min(next_t, tt)
            for tt in avail_l.values():
                if tt > t + eps:
                    next_t = min(next_t, tt)

            remaining = [i for i in I if i not in scheduled and is_pred_done(i)]
            if remaining:
                R2 = [max(_ready_time(i, job_of, data, C), float(start_time_floor)) for i in remaining]
                for rt in R2:
                    if rt > t + eps:
                        next_t = min(next_t, rt)

            if next_t == float("inf"):
                raise RuntimeError("Stuck: cannot advance time (check data feasibility).")
            t = next_t

    # objectives
    C_weld = {}
    C_final = {}
    T = {}
    for j in J:
        cwj = 0.0
        for op in O_j[j]:
            cwj = max(cwj, float(C[int(op)]))
        C_weld[int(j)] = cwj

        cf = cwj + float(g_j[j]) * float(t_grind_j[j]) + float(p_flag_j[j]) * float(t_paint_j[j])
        C_final[int(j)] = float(cf)
        T[int(j)] = max(0.0, float(cf) - float(d_j[j]))

    Tmax = max(T.values()) if T else 0.0
    Cmax = max(C_final.values()) if C_final else 0.0

    schedule = []
    for i in sorted(I, key=lambda x: (S.get(x, 0.0), C.get(x, 0.0), x)):
        # NOTE: am/al should exist for all scheduled ops, but be safe:
        schedule.append({
            "op_id": int(i),
            "op_label": str(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(S[i]),
            "finish": float(C[i]),
            "machine": int(am[i]) if i in am else None,
            "station": int(al[i]) if i in al else None,
        })

    obj = {
        "T_max": float(Tmax),
        "C_max": float(Cmax),
        "C_final": {int(j): float(C_final[j]) for j in C_final},
        "T": {int(j): float(T[j]) for j in T},
    }
    return schedule, obj


# ==========================================================
#  PUBLIC: solve_reschedule (urgent + outages)
# ==========================================================

def solve_reschedule_open_source(
    data_base: Dict[str, Any],
    old_solution: Dict[str, Any],
    urgent_payload: Optional[dict],
    unavailable_machines: Optional[List[int]] = None,
    unavailable_stations: Optional[List[int]] = None,
    mode: str = "continue",   # "continue" tercih 1
) -> Dict[str, Any]:
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []
    mode = (mode or "continue").strip().lower()

    # ✅ CRITICAL: normalize scenario data (JSON-safe -> Python-safe)
    data_base = normalize_data(data_base)

    old_solution = normalize_old_solution(old_solution)

    if "plan_start_iso" not in old_solution:
        raise ValueError("baseline_solution.json is missing plan_start_iso.")

    plan_calendar = old_solution.get("plan_calendar") or {"utc_offset": "+03:00"}
    t0 = compute_t0_from_plan_start(old_solution["plan_start_iso"], plan_calendar=plan_calendar)

    I0 = _to_int_list(data_base["I"])
    I_done, I_run, I_free = classify_ops_by_t0(I0, old_solution["S_old"], old_solution["C_old"], t0)

    # Apply urgent
    data1 = data_base
    if urgent_payload:
        data1 = add_urgent_job_from_payload(data_base, t0=t0, urgent_payload=urgent_payload)

    # Split remainders for running ops
    data2 = add_split_remainders_for_running_ops(data1, I_run, t0, old_solution)

    job_of = _job_of(data2)

    x_old = old_solution.get("x_old", {})
    y_old = old_solution.get("y_old", {})

    # Build fallback map from old_solution["schedule"] if present
    schedule_rows = old_solution.get("schedule", []) or []
    sch_machine = {}
    sch_station = {}
    for r in schedule_rows:
        try:
            op_id = int(r.get("op_id"))
        except Exception:
            continue
        if "machine" in r and r["machine"] is not None:
            sch_machine[op_id] = int(r["machine"])
        if "station" in r and r["station"] is not None:
            sch_station[op_id] = int(r["station"])

    badM = set(int(m) for m in unavailable_machines)
    badL = set(int(l) for l in unavailable_stations)

    def old_machine(i: int) -> Optional[int]:
        # prefer x_old
        for (ii, m), v in x_old.items():
            if int(ii) == int(i) and int(v) == 1:
                return int(m)
        # fallback schedule
        if int(i) in sch_machine:
            return int(sch_machine[int(i)])
        return None

    def old_station(i: int) -> Optional[int]:
        for (ii, l), v in y_old.items():
            if int(ii) == int(i) and int(v) == 1:
                return int(l)
        if int(i) in sch_station:
            return int(sch_station[int(i)])
        return None

    # Fixed ops = done + running (running possibly cut)
    fixed: Dict[int, dict] = {}

    for i in I_done:
        i = int(i)
        m0 = old_machine(i)
        l0 = old_station(i)

        # final fallback if still None:
        if m0 is None:
            m0 = int(data2["M_i"][i][0])
        if l0 is None:
            l0 = int(data2["L_i"][i][0])

        fixed[i] = {
            "op_id": i,
            "op_label": str(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(old_solution["S_old"][i]),
            "finish": float(old_solution["C_old"][i]),
            "machine": int(m0),
            "station": int(l0),
        }

    keep_decisions: Dict[int, int] = {}

    for i in I_run:
        i = int(i)
        m0 = old_machine(i)
        l0 = old_station(i)

        if m0 is None:
            m0 = int(data2["M_i"][i][0])
        if l0 is None:
            l0 = int(data2["L_i"][i][0])

        start_i = float(old_solution["S_old"][i])
        finish_i = float(old_solution["C_old"][i])

        broken = ((m0 is not None and int(m0) in badM) or (l0 is not None and int(l0) in badL))

        if broken:
            finish_i = float(t0)   # outage => cut
            keep_decisions[i] = 0
        else:
            # open-source version: default keep=1
            keep_decisions[i] = 1

        fixed[i] = {
            "op_id": i,
            "op_label": str(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(start_i),
            "finish": float(finish_i),
            "machine": int(m0),
            "station": int(l0),
        }

    # Schedule remaining with GT+ATC
    sched, obj = heuristic_reschedule_gt_atc(
        data=data2,
        fixed_ops=fixed,
        start_time_floor=float(t0),
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        k1=2.0,
    )

    # Urgent info
    urgent_info = None
    if urgent_payload:
        uj = int(data2.get("urgent_job_id"))
        uops = [int(x) for x in data2.get("urgent_ops", [])]

        C_by_op = {int(r["op_id"]): float(r["finish"]) for r in sched}
        Cw_u = 0.0
        for op in data2["O_j"][uj]:
            Cw_u = max(Cw_u, C_by_op[int(op)])

        pflag_map = data2.get("p_flag_j", data2.get("p_j", {})) or {}
        Cf_u = (
            float(Cw_u)
            + float(data2["g_j"][uj]) * float(data2["t_grind_j"][uj])
            + float(pflag_map.get(uj, 0)) * float(data2["t_paint_j"][uj])
        )
        T_u = max(0.0, Cf_u - float(data2["d_j"][uj]))
        urgent_info = {
            "job_id": uj,
            "ops": uops,
            "C_final": float(Cf_u),
            "T": float(T_u),
            "d": float(data2["d_j"][uj]),
        }

    return {
        "t0": float(t0),
        "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free},
        "objective": {"T_max": float(obj["T_max"]), "C_max": float(obj["C_max"])},
        "urgent": urgent_info,
        "keep_decisions": {int(k): int(v) for k, v in keep_decisions.items()},
        "schedule": sched,
        "changed_ops": [],
        "note": "Open-source heuristic reschedule (GT+ATC)."
    }
