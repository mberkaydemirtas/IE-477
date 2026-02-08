#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
solver_core.py

Heuristic-based scheduling and rescheduling core.

Both baseline and reschedule are solved via:
    HeuristicBaseModel.run_heuristic
"""

from datetime import datetime, timezone, timedelta, time as dtime
from typing import Any, Dict, List, Tuple, Optional

from HeuristicBaseModel import run_heuristic, HeuristicResult


# ==========================================================
#  TIME HELPERS
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

def _parse_iso_dt(iso_str: str) -> datetime:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def compute_t0_from_plan_start(
    plan_start_iso: str,
    plan_calendar: Optional[dict] = None,
    now_iso: Optional[str] = None
) -> float:

    plan_calendar = plan_calendar or {}
    utc_offset = plan_calendar.get("utc_offset", "+03:00")
    local_tz = _parse_utc_offset(utc_offset)

    shift_start_local = plan_calendar.get("shift_start_local", "08:00")
    workday_hours = float(plan_calendar.get("workday_hours", 8.0))
    workdays = set(int(x) for x in plan_calendar.get("workdays", [0, 1, 2, 3, 4]))

    start_utc = _parse_iso_dt(plan_start_iso)
    start_local = start_utc.astimezone(local_tz)

    now_utc = _parse_iso_dt(now_iso) if now_iso else datetime.now(timezone.utc)
    now_local = now_utc.astimezone(local_tz)

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
#  DATA NORMALIZATION HELPERS
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
    if p_im_raw is None:
        return {}
    out: Dict[Tuple[int, int], float] = {}

    if isinstance(p_im_raw, dict):
        sample_key = next(iter(p_im_raw.keys()), None)

        if isinstance(sample_key, tuple) and len(sample_key) == 2:
            for (i, m), v in p_im_raw.items():
                out[(int(i), int(m))] = float(v)
            return out

        if isinstance(sample_key, str) and "," in sample_key:
            for k, v in p_im_raw.items():
                kk = str(k).replace("(", "").replace(")", "").strip()
                a, b = [x.strip() for x in kk.split(",")]
                out[(int(a), int(b))] = float(v)
            return out

        if sample_key is not None and isinstance(p_im_raw.get(sample_key), dict):
            for i_key, inner in p_im_raw.items():
                i = int(i_key)
                for m_key, v in inner.items():
                    out[(i, int(m_key))] = float(v)
            return out

    return out

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
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

    if (not out.get("p_im")) and (out.get("p_i") is not None):
        p_i_raw = out.get("p_i") or {}
        if not isinstance(p_i_raw, dict):
            raise ValueError("p_i must be a dict like {op_id: processing_time}")

        p_i = {int(k): float(v) for k, v in p_i_raw.items()}

        p_im: Dict[Tuple[int, int], float] = {}
        for i in out["I"]:
            ii = int(i)
            if ii not in p_i:
                raise ValueError(f"Missing p_i for operation {ii}")
            feas_m = out["M_i"].get(ii, [])
            if not feas_m:
                raise ValueError(f"Missing M_i for operation {ii}")
            for m in feas_m:
                p_im[(ii, int(m))] = float(p_i[ii])

        out["p_im"] = p_im

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

    # optional default ptime for urgent fallback
    if "default_processing_time_hours" not in out:
        out["default_processing_time_hours"] = 1.0

    return out

def normalize_old_solution(old_solution: dict) -> dict:
    out = dict(old_solution)

    for k in ["S_old", "C_old"]:
        if k in out and isinstance(out[k], dict):
            out[k] = {int(kk): float(vv) for kk, vv in out[k].items()}

    for k in ["x_old", "y_old"]:
        if k in out and isinstance(out[k], dict):
            conv = {}
            for kk, vv in out[k].items():
                if isinstance(kk, tuple) and len(kk) == 2:
                    i, m = kk
                    conv[(int(i), int(m))] = int(vv)
                else:
                    s = str(kk).replace("(", "").replace(")", "").strip()
                    a, b = [x.strip() for x in s.split(",")]
                    conv[(int(a), int(b))] = int(vv)
            out[k] = conv

    return out


# ==========================================================
#  BASELINE SOLVE
# ==========================================================

def _op_to_job(O_j: Dict[int, List[int]]) -> Dict[int, int]:
    mp = {}
    for j, ops in O_j.items():
        for op in ops:
            mp[int(op)] = int(j)
    return mp

def solve_baseline(
    data: Dict[str, Any],
    plan_start_iso: str,
    plan_calendar: Optional[dict] = None,
    k1: float = 2.0
) -> Dict[str, Any]:
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

    schedule = []
    job_of = _op_to_job(data["O_j"])
    for i in sorted(data["I"], key=lambda ii: (res.S[int(ii)], res.C[int(ii)], int(ii))):
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

    S_old = {int(i): float(res.S[int(i)]) for i in data["I"]}
    C_old = {int(i): float(res.C[int(i)]) for i in data["I"]}

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
#  RESCHEDULE HELPERS
# ==========================================================

def classify_ops_by_t0(I: List[int], S_old: Dict[int, float], C_old: Dict[int, float], t0: float):
    """
    Robust classification:
    - If an op is missing in baseline (no S_old/C_old), treat as FREE (new/unseen op).
    """
    done = []
    run = []
    free = []

    for i in I:
        ii = int(i)

        # baseline doesn't know this op -> treat as not started
        if ii not in S_old or ii not in C_old:
            free.append(ii)
            continue

        s = float(S_old[ii])
        c = float(C_old[ii])

        if c <= t0 + 1e-9:
            done.append(ii)
        elif (s < t0 - 1e-9) and (c > t0 + 1e-9):
            run.append(ii)
        else:
            free.append(ii)

    return done, run, free


# ==========================================================
#  URGENT JOB INJECTION
# ==========================================================

def add_urgent_job_from_payload(data: Dict[str, Any], t0: float, urgent_payload: dict) -> Dict[str, Any]:
    """
    Accepts BOTH formats:
    - old: ops[*].processing_time_by_machine / feasible_machines / feasible_stations
    - new: ops[*].cycleTime (minutes), machineCandidates, stationCandidates, stationSizeRequirement
    """
    uj = urgent_payload.get("urgent_job", urgent_payload)

    data = normalize_data(data)

    J = list(data["J"])
    I = list(data["I"])
    O_j = dict(data["O_j"])
    L = list(data["L"])
    M = list(data["M"])

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
        if op_id is not None:
            try:
                op_id = int(op_id)
            except Exception:
                op_id = None

        if (op_id is None) or (op_id in I) or (op_id in new_ops):
            max_op += 1
            op_id = max_op

        op["op_id"] = op_id
        new_ops.append(op_id)

    J2 = J + [job_id]
    I2 = I + new_ops
    O_j2 = dict(O_j)
    O_j2[job_id] = list(new_ops)

    default_pt_hours = float(data.get("default_processing_time_hours", 1.0))

    for op_def in ops_payload:
        op_id = int(op_def["op_id"])

        # ----- stations -----
        feasL = op_def.get("feasible_stations", None)
        if feasL is None:
            feasL = op_def.get("stationCandidates", None)
        if feasL is None:
            feasL = L
        L_i[op_id] = [int(l) for l in feasL] if feasL else list(L)
        if not L_i[op_id]:
            L_i[op_id] = list(L)

        # ----- machines -----
        feasM_user = op_def.get("feasible_machines", None)
        if feasM_user is None:
            feasM_user = op_def.get("machineCandidates", None)
        if feasM_user is None:
            feasM_final = list(M)
        else:
            feasM_final = sorted({int(x) for x in feasM_user})

        if not feasM_final:
            raise ValueError(f"urgent op {op_id}: machine candidates empty")
        M_i[op_id] = list(feasM_final)

        # ----- ptime -----
        pt = op_def.get("processing_time_by_machine", None)
        if not pt:
            cycle_time = op_def.get("cycleTime", None)  # minutes
            if cycle_time is not None:
                try:
                    pt_hours = float(cycle_time) / 60.0
                except Exception:
                    pt_hours = default_pt_hours
            else:
                pt_hours = default_pt_hours

            for m in M_i[op_id]:
                p_im[(op_id, int(m))] = float(pt_hours)
        else:
            if not isinstance(pt, dict):
                raise ValueError(f"urgent op {op_id}: processing_time_by_machine must be dict")
            for m in M_i[op_id]:
                val = pt.get(str(m), pt.get(m, None))
                if val is None:
                    raise ValueError(f"urgent op {op_id}: missing ptime for machine {m}")
                p_im[(op_id, int(m))] = float(val)

        # ----- big/small requirement -----
        ssr = (op_def.get("stationSizeRequirement", None) or "").strip().upper()
        if ssr == "BIG":
            beta_i[op_id] = 1
        elif ssr in ("SMALL", "ANY", ""):
            beta_i[op_id] = 0
        else:
            beta_i[op_id] = 1 if bool(op_def.get("beta_big_station_required", False)) else 0

        Pred_i[op_id] = []

    edges = uj.get("precedence_edges", []) or []
    if (not edges) and len(new_ops) >= 2:
        edges = [[new_ops[k], new_ops[k + 1]] for k in range(len(new_ops) - 1)]

    for a, b in edges:
        a = int(a)
        b = int(b)
        if a not in I2 or b not in I2:
            continue
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
        "urgent_ops": new_ops,
        "urgent_weight": float(uj.get("urgent_weight", 25.0)),
    })
    return normalize_data(out)


# ==========================================================
#  RUNNING OPS REMAINDERS
# ==========================================================

def add_split_remainders_for_running_ops(
    data: Dict[str, Any],
    I_run_broken: List[int],
    t0: float,
    old_solution: Dict[str, Any]
) -> Dict[str, Any]:
    if not I_run_broken:
        out = dict(data)
        out["rem_map"] = {}
        return normalize_data(out)

    data = normalize_data(data)
    old_solution = normalize_old_solution(old_solution)

    J, I, O_j = data["J"], data["I"], data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    beta_i = data["beta_i"]

    S_old = old_solution["S_old"]
    x_old = old_solution.get("x_old", {})

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
        # NOTE: your x_old currently stored as {"i,m": 1} strings in baseline.
        for kk, v in (x_old or {}).items():
            if int(v) != 1:
                continue
            if isinstance(kk, tuple) and len(kk) == 2:
                ii, m = kk
                if int(ii) == int(i):
                    return int(m)
            else:
                s = str(kk)
                if "," in s:
                    a, b = s.split(",")
                    if int(a) == int(i):
                        return int(b)
        return None

    for i in I_run_broken:
        i = int(i)

        i_rem = next_op
        next_op += 1
        rem_map[i] = i_rem

        I2.append(i_rem)

        j = op_to_job.get(i, None)
        if j is not None and j in O_j2:
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
        beta_i2[i_rem] = int(beta_i2.get(i, 0))

        for k in list(Pred_i2.keys()):
            if int(k) == int(i_rem):
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
    return normalize_data(out)


# ==========================================================
#  RESCHEDULE SOLVE
# ==========================================================

def solve_reschedule(
    data_base: Dict[str, Any],
    old_solution: Dict[str, Any],
    urgent_payload: Optional[dict] = None,
    unavailable_machines: Optional[List[int]] = None,
    unavailable_stations: Optional[List[int]] = None,
    mode: str = "continue",
    k1: float = 2.0,
    t_now_iso: Optional[str] = None,
) -> Dict[str, Any]:

    unavailable_machines = [int(x) for x in (unavailable_machines or [])]
    unavailable_stations = [int(x) for x in (unavailable_stations or [])]
    mode = (mode or "continue").strip().lower()

    data_base = normalize_data(data_base)
    old_solution = normalize_old_solution(old_solution)

    if "plan_start_iso" not in old_solution:
        raise ValueError("baseline_solution is missing plan_start_iso")

    plan_calendar = old_solution.get("plan_calendar") or {"utc_offset": "+03:00"}

    t0 = compute_t0_from_plan_start(
        old_solution["plan_start_iso"],
        plan_calendar=plan_calendar,
        now_iso=t_now_iso,
    )

    I0 = _to_int_list(data_base["I"])
    S_old = old_solution["S_old"]
    C_old = old_solution["C_old"]
    I_done, I_run, I_free = classify_ops_by_t0(I0, S_old, C_old, t0)

    data1 = data_base
    if urgent_payload:
        data1 = add_urgent_job_from_payload(data1, t0=t0, urgent_payload=urgent_payload)

    sch_machine: Dict[int, int] = {}
    sch_station: Dict[int, int] = {}

    for r in (old_solution.get("schedule", []) or []):
        try:
            op_id = int(r.get("op_id"))
        except Exception:
            continue
        if r.get("machine", None) is not None:
            sch_machine[op_id] = int(r["machine"])
        if r.get("station", None) is not None:
            sch_station[op_id] = int(r["station"])

    # x_old/y_old stored as "i,m" strings -> already ok in add_split_remainders_for_running_ops
    # keep only schedule dicts above

    badM = set(unavailable_machines)
    badL = set(unavailable_stations)

    preempt_for_urgent = True
    if urgent_payload and isinstance(urgent_payload, dict):
        if "preempt_for_urgent" in urgent_payload:
            preempt_for_urgent = bool(urgent_payload.get("preempt_for_urgent"))
        elif isinstance(urgent_payload.get("urgent_job"), dict) and "preempt_for_urgent" in urgent_payload["urgent_job"]:
            preempt_for_urgent = bool(urgent_payload["urgent_job"].get("preempt_for_urgent"))

    urgent_machines: set = set()
    if urgent_payload and preempt_for_urgent:
        uj = urgent_payload.get("urgent_job", urgent_payload) if isinstance(urgent_payload, dict) else {}
        for op in uj.get("ops", []) or []:
            pt = op.get("processing_time_by_machine", {}) or {}
            if pt:
                urgent_machines |= set(int(k) for k in pt.keys())
            else:
                mc = op.get("feasible_machines", None) or op.get("machineCandidates", None) or []
                urgent_machines |= set(int(x) for x in mc)

    fixed_ops: Dict[int, Dict[str, Any]] = {}
    keep_decisions: Dict[int, int] = {}
    I_run_broken: List[int] = []

    for i in I_done:
        i = int(i)
        fixed_ops[i] = {
            "start": float(S_old[i]),
            "finish": float(C_old[i]),
            "machine": sch_machine.get(i, None),
            "station": sch_station.get(i, None),
        }

    for i in I_run:
        i = int(i)
        m0 = sch_machine.get(i, None)
        l0 = sch_station.get(i, None)

        broken_by_failure = ((m0 is not None and int(m0) in badM) or (l0 is not None and int(l0) in badL))
        broken_by_urgent = (preempt_for_urgent and urgent_machines and (m0 is not None and int(m0) in urgent_machines))

        start_i = float(S_old[i])
        finish_i = float(C_old[i])

        if broken_by_failure or broken_by_urgent:
            finish_i = float(t0)
            keep_decisions[i] = 0
            I_run_broken.append(i)
        else:
            keep_decisions[i] = 1

        fixed_ops[i] = {
            "start": float(start_i),
            "finish": float(finish_i),
            "machine": m0,
            "station": l0,
        }

    data2 = add_split_remainders_for_running_ops(data1, I_run_broken, t0, old_solution)
    job_of = _op_to_job(data2["O_j"])

    res: HeuristicResult = run_heuristic(
        data2,
        k1=float(k1),
        fixed_ops=fixed_ops,
        start_time_floor=float(t0),
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
    )

    rem_map = data2.get("rem_map", {}) or {}
    inv_rem = {int(v): int(k) for k, v in rem_map.items()}

    schedule = []
    for i in sorted(data2["I"], key=lambda ii: (res.S[int(ii)], res.C[int(ii)], int(ii))):
        i = int(i)
        op_label = f"{inv_rem[i]} (cont)" if i in inv_rem else str(i)
        schedule.append({
            "op_id": i,
            "op_label": op_label,
            "job_id": int(job_of.get(i, -1)),
            "start": float(res.S[i]),
            "finish": float(res.C[i]),
            "machine": int(res.assign_machine.get(i)) if i in res.assign_machine else None,
            "station": int(res.assign_station.get(i)) if i in res.assign_station else None,
        })

    urgent_info = None
    if urgent_payload and "urgent_job_id" in data2:
        ujid = int(data2["urgent_job_id"])
        Cf_u = float(res.C_final.get(ujid, 0.0))
        Tu = float(res.T.get(ujid, 0.0))
        urgent_info = {"job_id": ujid, "C_final": Cf_u, "T": Tu, "d": float(data2["d_j"][ujid])}

    return {
        "t0": float(t0),
        "t_now_iso": t_now_iso,
        "mode": mode,
        "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free, "I_run_broken": I_run_broken},
        "objective": {"T_max": float(res.T_max), "C_max": float(res.C_max)},
        "urgent": urgent_info,
        "keep_decisions": {int(k): int(v) for k, v in keep_decisions.items()},
        "rem_map": {int(k): int(v) for k, v in rem_map.items()},
        "schedule": schedule,
        "note": "Reschedule solved by same engine: HeuristicBaseModel.run_heuristic"
    }

def solve_reschedule_open_source(*args, **kwargs):
    return solve_reschedule(*args, **kwargs)
