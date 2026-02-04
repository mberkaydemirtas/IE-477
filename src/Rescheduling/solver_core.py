#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
solver_core.py (Heuristic-based, single-engine)

✅ Baseline + Reschedule are BOTH solved by the SAME heuristic:
    HeuristicBaseModel.run_heuristic

Key ideas:
- baseline: run_heuristic(data) -> baseline_solution.json
- reschedule: compute t0 (working-hours since plan_start_iso)
    - done ops: freeze [S_old, C_old] + keep machine/station
    - running ops: if assigned resource becomes unavailable => cut at t0 (preempt)
                  else keep as fixed (continue)
    - free ops: cannot start before t0 (start_time_floor)
    - urgent job: optional payload injected into data
    - running ops get remainder ops (split) always; if kept, remainder can be zeroed by fixed ops logic upstream (optional)

Scenario JSON:
- `data` is OPTIONAL. If missing, default base data is used.
- `overrides` is OPTIONAL (partial updates to base data).
- urgent job payload example supported (your format).
"""

import math
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Any, Dict, List, Tuple, Optional

from HeuristicBaseModel import run_heuristic, HeuristicResult


# ==========================================================
#  TIME HELPERS (working-hours based t0, no tzdata dependency)
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

def compute_t0_from_plan_start(plan_start_iso: str, plan_calendar: Optional[dict] = None) -> float:
    """
    t0 = elapsed since plan_start_iso measured in working hours.

    plan_start_iso: UTC ISO string (timezone-aware recommended)
    plan_calendar: {
        "utc_offset": "+03:00",
        "shift_start_local": "08:00",
        "workday_hours": 8.0,
        "workdays": [0,1,2,3,4]  # Mon-Fri
    }
    """
    plan_calendar = plan_calendar or {}
    utc_offset = plan_calendar.get("utc_offset", "+03:00")
    local_tz = _parse_utc_offset(utc_offset)

    shift_start_local = plan_calendar.get("shift_start_local", "08:00")
    workday_hours = float(plan_calendar.get("workday_hours", 8.0))

    workdays = set(int(x) for x in plan_calendar.get("workdays", [0, 1, 2, 3, 4]))

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
            left = max(start_local, s) if d == start_local.date() else s
            right = e
            if right > left:
                total += (right - left).total_seconds() / 3600.0
        d = d + timedelta(days=1)

    if today.weekday() in workdays:
        s, e = shift_window_for_date(today)
        left = max(s, start_local) if today == start_local.date() else s
        right = min(now_local, e)
        if right > left:
            total += (right - left).total_seconds() / 3600.0

    return float(total)


# ==========================================================
#  JSON / KEY NORMALIZATION HELPERS
# ==========================================================

def _to_int_list(xs) -> List[int]:
    return [int(x) for x in (xs or [])]

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
    Supports JSON formats:
    A) {"1,2": 3.5, "1,3": 4.0}
    B) {"1": {"2": 3.5, "3": 4.0}}
    C) {(1,2): 3.5}  (already tuple keys - python dict)
    """
    if p_im_raw is None:
        return {}
    out: Dict[Tuple[int, int], float] = {}

    if isinstance(p_im_raw, dict):
        sample_key = next(iter(p_im_raw.keys()), None)

        # C
        if isinstance(sample_key, tuple) and len(sample_key) == 2:
            for (i, m), v in p_im_raw.items():
                out[(int(i), int(m))] = float(v)
            return out

        # A
        if isinstance(sample_key, str) and "," in sample_key:
            for k, v in p_im_raw.items():
                kk = str(k).replace("(", "").replace(")", "").strip()
                a, b = [x.strip() for x in kk.split(",")]
                out[(int(a), int(b))] = float(v)
            return out

        # B
        if isinstance(p_im_raw.get(sample_key), dict):
            for i_key, inner in p_im_raw.items():
                i = int(i_key)
                for m_key, v in inner.items():
                    out[(i, int(m_key))] = float(v)
            return out

    return out

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes scenario `data` safe for int-key access and tuple p_im indexing.
    """
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    out = dict(data)

    out["J"] = _to_int_list(out.get("J"))
    out["I"] = _to_int_list(out.get("I"))
    out["M"] = _to_int_list(out.get("M"))
    out["L"] = _to_int_list(out.get("L"))

    out["O_j"] = _dict_int_keys_list_int_vals(out.get("O_j", {}))
    out["M_i"] = _dict_int_keys_list_int_vals(out.get("M_i", {}))
    out["L_i"] = _dict_int_keys_list_int_vals(out.get("L_i", {}))
    out["Pred_i"] = _dict_int_keys_list_int_vals(out.get("Pred_i", {}))

    out["r_j"] = _dict_int_keys_float_vals(out.get("r_j", {}))
    out["d_j"] = _dict_int_keys_float_vals(out.get("d_j", {}))
    out["g_j"] = _dict_int_keys_int_vals(out.get("g_j", {}))

    pflag = out.get("p_flag_j", out.get("p_j", {}))
    out["p_flag_j"] = _dict_int_keys_int_vals(pflag or {})

    out["t_grind_j"] = _dict_int_keys_float_vals(out.get("t_grind_j", {}))
    out["t_paint_j"] = _dict_int_keys_float_vals(out.get("t_paint_j", {}))
    out["beta_i"] = _dict_int_keys_int_vals(out.get("beta_i", {}))

    out["L_small"] = _to_int_list(out.get("L_small", []))
    if "L_big" in out:
        out["L_big"] = _to_int_list(out.get("L_big", []))

    out["p_im"] = _normalize_p_im(out.get("p_im"))

    # Fill missing job keys safely
    for j in out["J"]:
        if j not in out["p_flag_j"]:
            out["p_flag_j"][j] = 0
        if j not in out["g_j"]:
            out["g_j"][j] = 0
        if j not in out["r_j"]:
            out["r_j"][j] = 0.0
        if j not in out["d_j"]:
            out["d_j"][j] = 1e9
        if j not in out["t_grind_j"]:
            out["t_grind_j"][j] = 0.0
        if j not in out["t_paint_j"]:
            out["t_paint_j"][j] = 0.0

    return out


def normalize_old_solution(old_solution: dict) -> dict:
    out = dict(old_solution)

    for k in ["S_old", "C_old"]:
        if k in out and isinstance(out[k], dict):
            out[k] = {int(kk): float(vv) for kk, vv in out[k].items()}

    # x_old/y_old can be stored as {"i,m": 1}
    for k in ["x_old", "y_old"]:
        if k in out and isinstance(out[k], dict):
            conv = {}
            for kk, vv in out[k].items():
                if isinstance(kk, str):
                    s = kk.replace("(", "").replace(")", "").strip()
                    a, b = [x.strip() for x in s.split(",")]
                    conv[(int(a), int(b))] = int(vv)
                else:
                    i, m = kk
                    conv[(int(i), int(m))] = int(vv)
            out[k] = conv

    return out


# ==========================================================
#  DEFAULT BASE DATA (fallback if scenario has no "data")
# ==========================================================

def make_base_data(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Default base dataset: 8 jobs, 30 operations (similar to your earlier large example),
    compatible with HeuristicBaseModel.

    `overrides` may partially override keys (e.g., r_j, d_j, p_im, etc).
    """
    overrides = overrides or {}

    J = [1, 2, 3, 4, 5, 6, 7, 8]
    I = list(range(1, 31))

    O_j = {
        1: [1, 2, 3],
        2: [4, 5, 6, 7],
        3: [8, 9, 10, 11, 12],
        4: [13, 14],
        5: [15, 16, 17, 18],
        6: [19, 20, 21],
        7: [22, 23, 24, 25, 26],
        8: [27, 28, 29, 30],
    }

    M = list(range(1, 13))
    # machine types: 1..3 TIG (type=1), 4..12 MAG (type=2)
    machine_type = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2}

    # odd ops => TIG, even ops => MAG
    K_i = {i: ([1] if (i % 2 == 1) else [2]) for i in I}
    M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

    L = list(range(1, 10))
    L_i = {i: L[:] for i in I}
    L_big = [1, 2, 3]
    L_small = [4, 5, 6, 7, 8, 9]

    Pred_i = {i: [] for i in I}
    Pred_i[2] = [1]
    Pred_i[3] = [1]
    Pred_i[5] = [4]
    Pred_i[6] = [4]
    Pred_i[7] = [5]
    Pred_i[9] = [8]
    Pred_i[10] = [8]
    Pred_i[11] = [9, 10]
    Pred_i[12] = [11]
    Pred_i[14] = [13]
    Pred_i[16] = [15]
    Pred_i[17] = [15]
    Pred_i[18] = [16]
    Pred_i[21] = [19, 20]
    Pred_i[23] = [22]
    Pred_i[24] = [22]
    Pred_i[25] = [23]
    Pred_i[26] = [24]

    # last op depends on all in job
    for j in J:
        ops = O_j[j]
        last = ops[-1]
        preds = set(Pred_i[last])
        for op in ops:
            if op != last:
                preds.add(op)
        Pred_i[last] = list(preds)

    p_im: Dict[Tuple[int, int], float] = {}
    for i in I:
        for m in M_i[i]:
            base = 3 + (i % 5)
            machine_add = (m - 1) * 0.5
            p_im[(i, m)] = float(base + machine_add)

    release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    r_j = {j: float(release_times[idx]) for idx, j in enumerate(J)}

    due_base = 30.0
    Iend = [O_j[j][-1] for j in J]
    d_i = {}
    for idx, i_last in enumerate(Iend):
        d_i[i_last] = due_base + 3.0 * idx
    d_j = {j: float(d_i[O_j[j][-1]]) for j in J}

    g_j = {j: (1 if idx % 2 == 0 else 0) for idx, j in enumerate(J)}
    p_flag_j = {j: (1 if idx in [0, 1, 2] else 0) for idx, j in enumerate(J)}

    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)
        t_paint_j[j] = 3.0 if p_flag_j[j] == 1 else 0.0

    # beta=1 only on job last ops by default
    beta_i = {i: (1 if i in Iend else 0) for i in I}

    base = {
        "J": J, "I": I, "O_j": O_j,
        "M": M, "L": L,
        "M_i": M_i, "L_i": L_i,
        "Pred_i": Pred_i,
        "p_im": p_im,
        "r_j": r_j, "d_j": d_j,
        "g_j": g_j, "p_flag_j": p_flag_j,
        "t_grind_j": t_grind_j, "t_paint_j": t_paint_j,
        "beta_i": beta_i,
        "L_small": L_small,
        "L_big": L_big,
        "machine_type": machine_type,
        "K_i": K_i,
    }

    # Apply shallow overrides
    for k, v in overrides.items():
        base[k] = v

    # Normalize to ensure int keys & tuple p_im
    return normalize_data(base)


# ==========================================================
#  BASELINE SOLVE
# ==========================================================

def solve_baseline(
    data: Dict[str, Any],
    plan_start_iso: str,
    plan_calendar: Optional[dict] = None,
    k1: float = 2.0
) -> Dict[str, Any]:
    """
    Returns a JSON-serializable baseline solution.
    """
    plan_calendar = plan_calendar or {"utc_offset": "+03:00"}

    data = normalize_data(data)
    res: HeuristicResult = run_heuristic(
        data,
        k1=float(k1),
        fixed_ops=None,
        start_time_floor=0.0,
        unavailable_machines=None,
        unavailable_stations=None,
    )

    # build schedule list
    schedule = []
    for i in sorted(data["I"], key=lambda ii: (res.S[int(ii)], res.C[int(ii)], int(ii))):
        i = int(i)
        schedule.append({
            "op_id": i,
            "op_label": str(i),
            "job_id": int(_op_to_job(data["O_j"]).get(i, -1)),
            "start": float(res.S[i]),
            "finish": float(res.C[i]),
            "machine": int(res.assign_machine.get(i)) if i in res.assign_machine else None,
            "station": int(res.assign_station.get(i)) if i in res.assign_station else None,
        })

    # store old solution maps
    S_old = {int(i): float(res.S[int(i)]) for i in data["I"]}
    C_old = {int(i): float(res.C[int(i)]) for i in data["I"]}

    # x_old/y_old as {"i,m":0/1} to keep JSON simple
    x_old = {}
    y_old = {}
    for i in data["I"]:
        i = int(i)
        m = res.assign_machine.get(i, None)
        l = res.assign_station.get(i, None)
        if m is not None:
            x_old[f"{i},{int(m)}"] = 1
        if l is not None:
            y_old[f"{i},{int(l)}"] = 1

    return {
        "plan_start_iso": plan_start_iso,
        "plan_calendar": plan_calendar,
        "objective": {"T_max": float(res.T_max), "C_max": float(res.C_max)},
        "S_old": S_old,
        "C_old": C_old,
        "x_old": x_old,
        "y_old": y_old,
        "schedule": schedule,
        "note": "Baseline solved by HeuristicBaseModel.run_heuristic"
    }


# ==========================================================
#  RESCHED HELPERS
# ==========================================================

def _op_to_job(O_j: Dict[int, List[int]]) -> Dict[int, int]:
    mp = {}
    for j, ops in O_j.items():
        for op in ops:
            mp[int(op)] = int(j)
    return mp

def classify_ops_by_t0(I: List[int], S_old: Dict[int, float], C_old: Dict[int, float], t0: float):
    done = [int(i) for i in I if float(C_old[int(i)]) <= t0 + 1e-9]
    run  = [int(i) for i in I if (float(S_old[int(i)]) < t0 - 1e-9) and (float(C_old[int(i)]) > t0 + 1e-9)]
    free = [int(i) for i in I if float(S_old[int(i)]) >= t0 - 1e-9]
    return done, run, free


# ==========================================================
#  URGENT JOB PAYLOAD (your JSON format)
# ==========================================================

def add_urgent_job_from_payload(data: Dict[str, Any], t0: float, urgent_payload: dict) -> Dict[str, Any]:
    """
    urgent_payload example:
    { "urgent_job": { job_id, release_time_mode, due_time_mode, due_time_hours, ops:[...], precedence_edges:[...], post_ops:{...} } }
    """
    uj = urgent_payload.get("urgent_job", urgent_payload)

    data = normalize_data(data)

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

    new_ops: List[int] = []
    max_op = max(I) if I else 0
    for op in ops_payload:
        op_id = op.get("op_id", None)
        if op_id is None:
            max_op += 1
            op_id = max_op
        op_id = int(op_id)
        if op_id in I or op_id in new_ops:
            raise ValueError(f"urgent op_id clashes: {op_id}")
        new_ops.append(op_id)

    # insert
    J2 = J + [job_id]
    I2 = I + new_ops
    O_j2 = dict(O_j)
    O_j2[job_id] = list(new_ops)

    for op_def in ops_payload:
        op_id = int(op_def["op_id"])

        feasL = op_def.get("feasible_stations", L)
        L_i[op_id] = [int(l) for l in feasL]

        pt = op_def.get("processing_time_by_machine", {})
        if not pt:
            raise ValueError(f"urgent op {op_id} missing processing_time_by_machine")

        # allow machine keys as strings
        pt_machines = sorted({int(mm) for mm in pt.keys()})

        feasM_user = op_def.get("feasible_machines", None)
        if feasM_user is None:
            feasM_final = pt_machines
        else:
            feasM_final = sorted(set(int(mm) for mm in feasM_user).intersection(pt_machines))

        if not feasM_final:
            raise ValueError(f"urgent op {op_id}: feasible_machines ∩ pt_machines is empty")

        M_i[op_id] = list(feasM_final)

        for m in M_i[op_id]:
            val = pt.get(str(m), pt.get(m, None))
            if val is None:
                raise ValueError(f"urgent op {op_id}: missing ptime for machine {m}")
            p_im[(op_id, int(m))] = float(val)

        beta_i[op_id] = 1 if bool(op_def.get("beta_big_station_required", False)) else 0
        Pred_i[op_id] = []

    edges = uj.get("precedence_edges", [])
    for a, b in edges:
        a = int(a); b = int(b)
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
    return normalize_data(out)


# ==========================================================
#  SPLIT REMAINDERS FOR RUNNING OPS
# ==========================================================

def add_split_remainders_for_running_ops(
    data: Dict[str, Any],
    I_run: List[int],
    t0: float,
    old_solution: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create remainder ops for running ops.
    - remainder p = remaining_time based on old machine (or fallback)
    - remainder depends on original op
    - successors also depend on remainder (so split is effective)
    - beta preserved (big-station requirement)
    """
    data = normalize_data(data)
    old_solution = normalize_old_solution(old_solution)

    J, I, O_j = data["J"], data["I"], data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    beta_i = data["beta_i"]

    S_old = old_solution["S_old"]
    x_old = old_solution.get("x_old", {})  # tuple-like map after normalize_old_solution

    next_op = max(I) + 1
    rem_map: Dict[int, int] = {}
    op_to_job = _op_to_job(O_j)

    I2 = list(I)
    O_j2 = {j: list(O_j[j]) for j in J}
    L_i2 = dict(L_i)
    M_i2 = dict(M_i)
    Pred_i2 = {k: list(Pred_i.get(k, [])) for k in I}
    p_im2 = dict(p_im)
    beta_i2 = dict(beta_i)

    def old_machine(i: int) -> Optional[int]:
        # normalized x_old keys are (i,m)
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
            m0 = int(M_i[i][0])

        full_p = float(p_im[(i, m0)])
        elapsed = max(0.0, float(t0 - float(S_old[i])))
        remaining = max(0.01, full_p - elapsed)

        for m in M_i2[i_rem]:
            p_im2[(i_rem, int(m))] = float(remaining)

        Pred_i2[i_rem] = [i]

        # preserve big-station requirement
        beta_i2[i_rem] = int(beta_i2.get(i, 0))

        # redirect successors
        for k in list(Pred_i2.keys()):
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
    return normalize_data(out)


# ==========================================================
#  RESCHEDULE SOLVE (single engine)
# ==========================================================

def solve_reschedule(
    data_base: Dict[str, Any],
    old_solution: Dict[str, Any],
    urgent_payload: Optional[dict] = None,
    unavailable_machines: Optional[List[int]] = None,
    unavailable_stations: Optional[List[int]] = None,
    mode: str = "continue",  # for now: continue only (keep running unless broken)
    k1: float = 2.0
) -> Dict[str, Any]:
    """
    Rescheduling with the SAME heuristic as baseline.

    - compute t0 using plan_start_iso + plan_calendar stored in baseline_solution
    - freeze done ops, freeze running ops (or cut if broken), schedule the rest after t0
    - add urgent job if provided
    - add remainder ops for running ops (split) to model preemption in a heuristic way
    """
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []
    mode = (mode or "continue").strip().lower()

    data_base = normalize_data(data_base)
    old_solution = normalize_old_solution(old_solution)

    if "plan_start_iso" not in old_solution:
        raise ValueError("baseline_solution is missing plan_start_iso")
    plan_calendar = old_solution.get("plan_calendar") or {"utc_offset": "+03:00"}

    t0 = compute_t0_from_plan_start(old_solution["plan_start_iso"], plan_calendar=plan_calendar)

    I0 = _to_int_list(data_base["I"])
    S_old = old_solution["S_old"]
    C_old = old_solution["C_old"]
    I_done, I_run, I_free = classify_ops_by_t0(I0, S_old, C_old, t0)

    # inject urgent job
    data1 = data_base
    if urgent_payload:
        data1 = add_urgent_job_from_payload(data1, t0=t0, urgent_payload=urgent_payload)

    # split remainders for running ops
    data2 = add_split_remainders_for_running_ops(data1, I_run, t0, old_solution)
    job_of = _op_to_job(data2["O_j"])

    # Build old machine/station from baseline schedule (most reliable)
    sch_machine = {}
    sch_station = {}
    for r in (old_solution.get("schedule", []) or []):
        try:
            op_id = int(r.get("op_id"))
        except Exception:
            continue
        if r.get("machine", None) is not None:
            sch_machine[op_id] = int(r["machine"])
        if r.get("station", None) is not None:
            sch_station[op_id] = int(r["station"])

    badM = set(int(m) for m in unavailable_machines)
    badL = set(int(l) for l in unavailable_stations)

    # Fixed ops dict for run_heuristic
    fixed_ops: Dict[int, Dict[str, Any]] = {}

    keep_decisions: Dict[int, int] = {}

    # DONE ops: fully fixed
    for i in I_done:
        i = int(i)
        fixed_ops[i] = {
            "start": float(S_old[i]),
            "finish": float(C_old[i]),
            "machine": sch_machine.get(i, None),
            "station": sch_station.get(i, None),
        }

    # RUNNING ops: continue unless broken
    for i in I_run:
        i = int(i)
        m0 = sch_machine.get(i, None)
        l0 = sch_station.get(i, None)

        broken = ((m0 is not None and int(m0) in badM) or (l0 is not None and int(l0) in badL))

        start_i = float(S_old[i])
        finish_i = float(C_old[i])

        if broken:
            finish_i = float(t0)  # cut at t0
            keep_decisions[i] = 0
        else:
            keep_decisions[i] = 1  # continue

        fixed_ops[i] = {
            "start": float(start_i),
            "finish": float(finish_i),
            "machine": m0,
            "station": l0,
        }

    # Solve with ONE heuristic engine
    res: HeuristicResult = run_heuristic(
        data2,
        k1=float(k1),
        fixed_ops=fixed_ops,
        start_time_floor=float(t0),
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
    )

    # Build schedule output list
    schedule = []
    for i in sorted(data2["I"], key=lambda ii: (res.S[int(ii)], res.C[int(ii)], int(ii))):
        i = int(i)
        schedule.append({
            "op_id": i,
            "op_label": str(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(res.S[i]),
            "finish": float(res.C[i]),
            "machine": int(res.assign_machine.get(i)) if i in res.assign_machine else None,
            "station": int(res.assign_station.get(i)) if i in res.assign_station else None,
        })

    # urgent summary (if exists)
    urgent_info = None
    if urgent_payload and "urgent_job_id" in data2:
        uj = int(data2["urgent_job_id"])
        # completion robust: use res.C_weld + post ops
        Cf_u = float(res.C_final.get(uj, 0.0))
        Tu = float(res.T.get(uj, 0.0))
        urgent_info = {"job_id": uj, "C_final": Cf_u, "T": Tu, "d": float(data2["d_j"][uj])}

    return {
        "t0": float(t0),
        "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free},
        "objective": {"T_max": float(res.T_max), "C_max": float(res.C_max)},
        "urgent": urgent_info,
        "keep_decisions": {int(k): int(v) for k, v in keep_decisions.items()},
        "schedule": schedule,
        "note": "Reschedule solved by same engine: HeuristicBaseModel.run_heuristic"
    }
# ----------------------------------------------------------
# Backward-compatible alias (older runners may import this)
# ----------------------------------------------------------
def solve_reschedule_open_source(*args, **kwargs):
    return solve_reschedule(*args, **kwargs)