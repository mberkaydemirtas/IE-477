#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gurobipy import Model, GRB, quicksum
from datetime import datetime, timezone, timedelta, time as dtime
import math
from typing import Dict, List, Tuple, Optional


# ==========================================================
#  TIME HELPERS (working-hours based t0, NO tzdata dependency)
# ==========================================================

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # days before today
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

    # today partial
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
#  SOLUTION EXTRACTION HELPERS
# ==========================================================

def extract_old_solution(I, x, y, S, C):
    S_old = {int(i): float(S[i].X) for i in I}
    C_old = {int(i): float(C[i].X) for i in I}

    x_old = {_tuplekey_to_str(i, m): int(round(x[i, m].X)) for (i, m) in x.keys()}
    y_old = {_tuplekey_to_str(i, l): int(round(y[i, l].X)) for (i, l) in y.keys()}

    return S_old, C_old, x_old, y_old


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


def extract_assignment_for_ops(I, M_i, L_i, x, y):
    chosen_m = {}
    chosen_l = {}

    for i in I:
        i = int(i)

        mm = None
        for m in M_i[i]:
            if (i, m) in x and x[i, m].X > 0.5:
                mm = int(m)
                break

        ll = None
        for l in L_i[i]:
            if (i, l) in y and y[i, l].X > 0.5:
                ll = int(l)
                break

        chosen_m[i] = mm
        chosen_l[i] = ll

    return chosen_m, chosen_l


# ==========================================================
#  HEURISTIC (ATCS list scheduling)
# ==========================================================

def _avg_min_p(I: List[int], M_i: Dict[int, List[int]], p_im: Dict[Tuple[int, int], float]) -> float:
    vals = []
    for i in I:
        mi = M_i.get(i, [])
        if not mi:
            continue
        vals.append(min(p_im[(i, m)] for m in mi))
    return float(sum(vals) / max(1, len(vals)))


def _atcs_priority(i: int, t: float, job_of: Dict[int, int],
                   d_j: Dict[int, float],
                   M_i: Dict[int, List[int]],
                   p_im: Dict[Tuple[int, int], float],
                   pbar: float,
                   k1: float = 2.0) -> float:
    """
    ATCS(i,t) = 1/pi * exp( - max{ d_job(i) - pi - t, 0 } / (k1 * pbar) )
    pi = min_m p_im
    """
    mi = M_i[i]
    pi = min(p_im[(i, m)] for m in mi)
    j = job_of[i]
    slack = max(d_j[j] - pi - t, 0.0)
    return (1.0 / max(1e-9, pi)) * math.exp(-slack / max(1e-9, (k1 * pbar)))


def heuristic_schedule(
    data: dict,
    fixed_ops: Optional[Dict[int, dict]] = None,
    start_time_floor: float = 0.0,
    k1: float = 2.0,
    unavailable_machines: Optional[List[int]] = None,
    unavailable_stations: Optional[List[int]] = None,
) -> Tuple[List[dict], dict]:
    """
    Produces a feasible schedule (no-overlap on machines/stations) by list scheduling.

    fixed_ops: {op_id: {"start","finish","machine","station","job_id","op_label"}}
              Fixed ops block their machine/station until finish time.

    start_time_floor: all non-fixed ops start >= this.

    unavailable_machines/stations:
      - Heuristic will NEVER assign new operations to these resources (t0 sonrası).

    Returns:
      (schedule_rows, objectives)
      objectives = {"T_max","C_max","C_final":{j:...},"T":{j:...}}
    """
    fixed_ops = fixed_ops or {}
    badM = set(int(x) for x in (unavailable_machines or []))
    badL = set(int(x) for x in (unavailable_stations or []))

    J = list(data["J"])
    I = [int(x) for x in data["I"]]
    O_j = data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    r_j = data["r_j"]
    d_j = data["d_j"]
    g_j = data["g_j"]
    p_j = data["p_j"]
    t_grind_j = data["t_grind_j"]
    t_paint_j = data["t_paint_j"]
    beta_i = data["beta_i"]
    L_big = set(data.get("L_big", []))
    L_small = set(data.get("L_small", []))

    job_of = op_to_job_map(O_j)

    mach_av = {}   # m -> time
    stat_av = {}   # l -> time

    S = {}
    C = {}
    chosen_m = {}
    chosen_l = {}

    schedule = []
    for op_id, row in fixed_ops.items():
        i = int(op_id)
        S[i] = float(row["start"])
        C[i] = float(row["finish"])
        chosen_m[i] = row.get("machine")
        chosen_l[i] = row.get("station")

        m = chosen_m[i]
        l = chosen_l[i]
        if m is not None:
            mach_av[int(m)] = max(mach_av.get(int(m), 0.0), C[i])
        if l is not None:
            stat_av[int(l)] = max(stat_av.get(int(l), 0.0), C[i])

        schedule.append({
            "op_id": i,
            "op_label": str(row.get("op_label", row.get("op_id", i))),
            "job_id": int(row.get("job_id", job_of.get(i, -1))),
            "start": float(S[i]),
            "finish": float(C[i]),
            "machine": (int(m) if m is not None else None),
            "station": (int(l) if l is not None else None),
        })

    def preds_done_time(i: int) -> float:
        preds = Pred_i.get(i, [])
        if not preds:
            return 0.0
        return max(C.get(int(h), 0.0) for h in preds)

    remaining = set(I) - set(int(x) for x in fixed_ops.keys())
    pbar = _avg_min_p(I, M_i, p_im)

    while remaining:
        eligible = []
        for i in list(remaining):
            j = job_of.get(i, None)
            if j is None:
                continue

            preds = Pred_i.get(i, [])
            if any(int(h) not in C for h in preds):
                continue

            ready = preds_done_time(i)
            ready = max(ready, float(start_time_floor))

            if i == O_j[j][0]:
                ready = max(ready, float(r_j[j]))

            eligible.append((i, ready))

        if not eligible:
            raise RuntimeError("Heuristic deadlock: no eligible operations. Check precedence (cycle?)")

        best_i = None
        best_score = -1e100
        best_ready = None
        for (i, ready) in eligible:
            score = _atcs_priority(
                i=i, t=ready,
                job_of=job_of, d_j=d_j, M_i=M_i, p_im=p_im, pbar=pbar, k1=k1
            )
            if score > best_score:
                best_score = score
                best_i = i
                best_ready = ready

        i = int(best_i)
        ready = float(best_ready)

        feasible_stations = list(L_i[i])

        if beta_i.get(i, 0) == 1:
            feasible_stations = [l for l in feasible_stations if int(l) in L_big]

        feasible_stations = [int(l) for l in feasible_stations if int(l) not in badL]
        if not feasible_stations:
            raise RuntimeError(f"Heuristic: op {i} has no feasible station after filtering (big/closed).")

        best = None  # (finish, start, m, l)
        for m in M_i[i]:
            m = int(m)
            if m in badM:
                continue

            p = float(p_im[(i, m)])
            m_ready = mach_av.get(m, 0.0)

            for l in feasible_stations:
                l = int(l)
                l_ready = stat_av.get(l, 0.0)

                s = max(ready, m_ready, l_ready)
                f = s + p

                cand = (f, s, m, l)
                if best is None or cand < best:
                    best = cand

        if best is None:
            raise RuntimeError(f"Heuristic: no feasible (machine,station) for op {i} after filtering.")

        f, s, m, l = best

        S[i] = float(s)
        C[i] = float(f)
        chosen_m[i] = int(m)
        chosen_l[i] = int(l)
        mach_av[m] = float(f)
        stat_av[l] = float(f)

        schedule.append({
            "op_id": int(i),
            "op_label": str(int(i)),
            "job_id": int(job_of.get(i, -1)),
            "start": float(s),
            "finish": float(f),
            "machine": int(m),
            "station": int(l),
        })

        remaining.remove(i)

    C_weld = {int(j): 0.0 for j in J}
    for j in J:
        for op in O_j[j]:
            C_weld[j] = max(C_weld[j], C[int(op)])

    C_final = {}
    T = {}
    for j in J:
        C_final[j] = float(C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j])
        T[j] = max(0.0, C_final[j] - float(d_j[j]))

    Tmax = max(T.values()) if T else 0.0
    Cmax = max(C_final.values()) if C_final else 0.0

    schedule.sort(key=lambda r: (float(r["start"]), float(r["finish"]), int(r["op_id"])))

    objectives = {
        "T_max": float(Tmax),
        "C_max": float(Cmax),
        "C_final": {int(j): float(C_final[j]) for j in C_final},
        "T": {int(j): float(T[j]) for j in T},
    }
    return schedule, objectives


# ==========================================================
#  DATA BUILD
# ==========================================================

def make_base_data(overrides: dict | None = None):
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

    machine_type = {m: 2 for m in M}
    machine_type[1] = 1
    machine_type[2] = 1

    K_i = {i: ([1] if (i % 2 == 1) else [2]) for i in I}
    M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

    L = list(range(1, 10))
    L_i = {i: L[:] for i in I}
    L_big = [1, 2, 3]
    L_small = [4, 5, 6, 7, 8, 9]

    Pred_i = {i: [] for i in I}
    Pred_i[2]  = [1]
    Pred_i[3]  = [1]
    Pred_i[5]  = [4]
    Pred_i[6]  = [4]
    Pred_i[7]  = [5]
    Pred_i[9]  = [8]
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

    for j in J:
        ops = O_j[j]
        last = ops[-1]
        preds = set(Pred_i[last])
        for op in ops:
            if op != last:
                preds.add(op)
        Pred_i[last] = list(preds)

    p_im = {}
    for i in I:
        for m in M_i[i]:
            base = 3 + (i % 5)
            machine_add = (m - 1) * 0.5
            p_im[(i, m)] = float(base + machine_add)

    if (1, 1) in p_im:
        p_im[(1, 1)] = 6.0

    release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    r_j = {j: release_times[idx] for idx, j in enumerate(J)}

    base_slack = 18.0
    slope = 1.2
    d_j = {j: float(r_j[j] + base_slack + slope * j) for j in J}

    g_j = {j: (1 if idx % 2 == 0 else 0) for idx, j in enumerate(J)}
    p_j = {j: (1 if idx in [0, 1, 2] else 0) for idx, j in enumerate(J)}

    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)
        t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

    Iend = [O_j[j][-1] for j in J]
    beta_i = {i: (1 if i in Iend else 0) for i in I}

    data = {
        "J": J, "I": I, "O_j": O_j,
        "M": M, "L": L,
        "machine_type": machine_type,
        "K_i": K_i,
        "M_i": M_i,
        "L_i": L_i,
        "L_big": L_big, "L_small": L_small,
        "Pred_i": Pred_i,
        "p_im": p_im,
        "r_j": r_j, "d_j": d_j,
        "g_j": g_j, "p_j": p_j,
        "t_grind_j": t_grind_j, "t_paint_j": t_paint_j,
        "beta_i": beta_i,
        "M_proc": 1000.0, "M_seq": 1000.0, "M_Lseq": 1000.0
    }

    for key, val in overrides.items():
        if key not in data:
            raise ValueError(f"Unknown override key in scenario: {key}")

        if isinstance(data[key], dict) and isinstance(val, dict):
            new_map = dict(data[key])
            for kk, vv in val.items():
                try:
                    ik = int(kk)
                except Exception:
                    ik = kk
                new_map[ik] = vv
            data[key] = new_map
        else:
            data[key] = val

    return data


# ==========================================================
#  URGENT JOB FROM PAYLOAD
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
    Pred_i = {k: list(v) for k, v in data["Pred_i"].items()}
    p_im = dict(data["p_im"])
    r_j = dict(data["r_j"])
    d_j = dict(data["d_j"])
    g_j = dict(data["g_j"])
    p_j = dict(data["p_j"])
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
            if val is None:
                raise ValueError(f"urgent op {op_id}: missing ptime for machine {m}")
            p_im[(op_id, m)] = float(val)

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
    p_j[job_id] = 1 if p_req else 0
    t_grind_j[job_id] = float(post.get("t_grind_hours", 0.0)) if g_req else 0.0
    t_paint_j[job_id] = float(post.get("t_paint_hours", 0.0)) if p_req else 0.0

    out = dict(data)
    out.update({
        "J": J2, "I": I2, "O_j": O_j2,
        "M_i": M_i, "L_i": L_i,
        "Pred_i": Pred_i, "p_im": p_im,
        "r_j": r_j, "d_j": d_j,
        "g_j": g_j, "p_j": p_j,
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
    x_old = old_solution["x_old"]

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
            if ii == int(i) and v == 1:
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
            p_im2[(i_rem, m)] = float(remaining)

        Pred_i2[i_rem] = [i]
        beta_i2[i_rem] = 0

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
#  MODEL BUILDER (objective is T_max only)
# ==========================================================

def build_model(name, data,
                unavailable_machines=None, unavailable_stations=None,
                freeze_done_ops=None,
                running_ops=None, mode="continue", t0=None,
                free_ops=None,
                old_solution=None):
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []
    freeze_done_ops = freeze_done_ops or []
    running_ops = running_ops or []
    free_ops = free_ops or []

    J = data["J"]; I = data["I"]; O_j = data["O_j"]
    M = data["M"]; L = data["L"]
    M_i = data["M_i"]; L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    r_j = data["r_j"]; d_j = data["d_j"]
    g_j = data["g_j"]; p_j = data["p_j"]
    t_grind_j = data["t_grind_j"]; t_paint_j = data["t_paint_j"]
    beta_i = data["beta_i"]
    L_small = data["L_small"]
    M_proc = data["M_proc"]; M_seq = data["M_seq"]; M_Lseq = data["M_Lseq"]
    rem_map = data.get("rem_map", {})

    remainder_ops = set(int(v) for v in rem_map.values()) if rem_map else set()
    rem_owner = {int(v): int(k) for k, v in rem_map.items()} if rem_map else {}

    model = Model(name)

    x = model.addVars([(int(i), int(m)) for i in I for m in M_i[int(i)]], vtype=GRB.BINARY, name="x")
    y = model.addVars([(int(i), int(l)) for i in I for l in L_i[int(i)]], vtype=GRB.BINARY, name="y")

    # --------- IMPORTANT: build zM/zL only for (a<b) and common resources ----------
    I_list = [int(v) for v in I]
    nI = len(I_list)

    zM_keys = []
    zL_keys = []
    for a in range(nI):
        for b in range(a + 1, nI):
            i = I_list[a]
            h = I_list[b]
            for m in set(M_i[i]).intersection(M_i[h]):
                zM_keys.append((i, h, int(m)))
            for l in set(L_i[i]).intersection(L_i[h]):
                zL_keys.append((i, h, int(l)))

    zM = model.addVars(zM_keys, vtype=GRB.BINARY, name="zM")
    zL = model.addVars(zL_keys, vtype=GRB.BINARY, name="zL")

    S = model.addVars(I_list, lb=0.0, vtype=GRB.CONTINUOUS, name="S")
    C = model.addVars(I_list, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    C_weld  = model.addVars([int(j) for j in J], lb=0.0, vtype=GRB.CONTINUOUS, name="C_weld")
    C_final = model.addVars([int(j) for j in J], lb=0.0, vtype=GRB.CONTINUOUS, name="C_final")

    T     = model.addVars([int(j) for j in J], lb=0.0, vtype=GRB.CONTINUOUS, name="T")
    T_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T_max")
    C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

    model.setObjective(T_max, GRB.MINIMIZE)

    running_set = set(int(i) for i in running_ops)
    badM = set(int(m) for m in unavailable_machines)
    badL = set(int(l) for l in unavailable_stations)

    # keep vars
    keep = None
    if old_solution is not None and running_ops:
        if t0 is None:
            raise ValueError("t0 must be set when running_ops exist")
        keep = model.addVars([int(i) for i in running_ops], vtype=GRB.BINARY, name="keep_running")

    # Assignment constraints (remainder activation depends on keep, in BOTH modes)
    for i in I_list:
        rhsM = 1
        rhsL = 1

        if i in remainder_ops:
            if keep is not None:
                orig = rem_owner[i]
                rhsM = 1 - keep[orig]
                rhsL = 1 - keep[orig]
            else:
                rhsM = 0
                rhsL = 0

        model.addConstr(quicksum(x[i, m] for m in M_i[i]) == rhsM, name=f"assign_M_{i}")
        model.addConstr(quicksum(y[i, l] for l in L_i[i]) == rhsL, name=f"assign_L_{i}")

    frozen_ops = set(int(i) for i in freeze_done_ops) | set(int(i) for i in running_ops)
    urgent_ops = set(int(i) for i in data.get("urgent_ops", []))
    free_ops_set = set(int(i) for i in free_ops)
    reschedulable_ops = (free_ops_set | urgent_ops | remainder_ops) - frozen_ops

    if unavailable_machines:
        for (ii, m) in list(x.keys()):
            if int(ii) in reschedulable_ops and int(m) in badM:
                model.addConstr(x[ii, m] == 0, name=f"unavailM_{ii}_{m}")

    if unavailable_stations:
        for (ii, l) in list(y.keys()):
            if int(ii) in reschedulable_ops and int(l) in badL:
                model.addConstr(y[ii, l] == 0, name=f"unavailL_{ii}_{l}")

    # PROCESSING TIME LINKING (running ops)
    if old_solution is not None and running_ops:
        S_old = old_solution["S_old"]
        C_old = old_solution["C_old"]
        x_old = old_solution["x_old"]
        y_old = old_solution["y_old"]

        def old_machine(i):
            for (ii, m), v in x_old.items():
                if ii == int(i) and v == 1:
                    return int(m)
            return None

        def old_station(i):
            for (ii, l), v in y_old.items():
                if ii == int(i) and v == 1:
                    return int(l)
            return None

        for i in [int(v) for v in running_ops]:
            m0 = old_machine(i)
            l0 = old_station(i)

            for m in M_i[i]:
                if (i, m) in x:
                    model.addConstr(x[i, m] == int(x_old.get((i, m), 0)), name=f"fix_run_x_{i}_{m}")
            for l in L_i[i]:
                if (i, l) in y:
                    model.addConstr(y[i, l] == int(y_old.get((i, l), 0)), name=f"fix_run_y_{i}_{l}")

            model.addConstr(S[i] == float(S_old[i]), name=f"fix_run_S_{i}")

            broken = ((m0 is not None and m0 in badM) or (l0 is not None and l0 in badL))
            if broken:
                model.addConstr(keep[i] == 0, name=f"force_preempt_unavail_{i}")
            else:
                if mode == "continue":
                    model.addConstr(keep[i] == 1, name=f"force_keep_continue_{i}")

            if m0 is None:
                m0 = M_i[i][0]

            full_p = float(p_im[(i, m0)])
            elapsed = max(0.01, float(t0 - float(S_old[i])))

            model.addGenConstrIndicator(keep[i], True,  C[i] == float(C_old[i]), name=f"ind_keepC_{i}")
            model.addGenConstrIndicator(keep[i], True,  C[i] - S[i] == full_p,   name=f"ind_keepDur_{i}")

            model.addGenConstrIndicator(keep[i], False, C[i] == float(t0),       name=f"ind_preemptC_{i}")
            model.addGenConstrIndicator(keep[i], False, C[i] - S[i] == elapsed,  name=f"ind_preemptDur_{i}")

            i_rem = rem_map.get(i, None)
            if i_rem is not None and int(i_rem) in I_list:
                i_rem = int(i_rem)
                model.addGenConstrIndicator(keep[i], True,  S[i_rem] == float(C_old[i]), name=f"ind_keepRemS_{i}")
                model.addGenConstrIndicator(keep[i], True,  C[i_rem] == float(C_old[i]), name=f"ind_keepRemC_{i}")
                model.addGenConstrIndicator(keep[i], False, S[i_rem] >= float(t0),       name=f"ind_preemptRemS_{i}")

    # Standard ptime constraints for NON-running ops
    for i in I_list:
        if i in running_set and old_solution is not None:
            continue
        for m in M_i[i]:
            pijm = float(p_im[(i, m)])
            model.addConstr(C[i] - S[i] >= pijm - M_proc * (1 - x[i, m]), name=f"ptime_lb_{i}_{m}")
            model.addConstr(C[i] - S[i] <= pijm + M_proc * (1 - x[i, m]), name=f"ptime_ub_{i}_{m}")

    # big station feasibility
    for i in I_list:
        if beta_i.get(i, 0) == 1:
            for l in L_small:
                if int(l) in set(int(v) for v in L_i[i]):
                    if (i, int(l)) in y:
                        model.addConstr(y[i, int(l)] == 0, name=f"small_forbid_{i}_{l}")

    # precedence
    for i in I_list:
        for h in Pred_i.get(i, []):
            model.addConstr(S[i] >= C[int(h)], name=f"prec_{h}_{i}")

    # release (job first op)
    for j in [int(v) for v in J]:
        first_op = int(O_j[j][0])
        model.addConstr(S[first_op] >= float(r_j[j]), name=f"release_{j}")

    # -------------------------
    # FIXED MACHINE SEQUENCING
    # -------------------------
    for (i, h, m) in zM.keys():
        # zM=1 -> i before h
        # zM=0 -> h before i
        model.addConstr(
            S[h] >= C[i]
                   - M_seq * (1 - zM[i, h, m])
                   - M_seq * (2 - x[i, m] - x[h, m]),
            name=f"no_overlapM_h_after_i_{i}_{h}_{m}"
        )
        model.addConstr(
            S[i] >= C[h]
                   - M_seq * (zM[i, h, m])
                   - M_seq * (2 - x[i, m] - x[h, m]),
            name=f"no_overlapM_i_after_h_{i}_{h}_{m}"
        )

    # -------------------------
    # FIXED STATION SEQUENCING
    # -------------------------
    for (i, h, l) in zL.keys():
        model.addConstr(
            S[h] >= C[i]
                   - M_Lseq * (1 - zL[i, h, l])
                   - M_Lseq * (2 - y[i, l] - y[h, l]),
            name=f"no_overlapL_h_after_i_{i}_{h}_{l}"
        )
        model.addConstr(
            S[i] >= C[h]
                   - M_Lseq * (zL[i, h, l])
                   - M_Lseq * (2 - y[i, l] - y[h, l]),
            name=f"no_overlapL_i_after_h_{i}_{h}_{l}"
        )

    # welding completion
    for j in [int(v) for v in J]:
        for op in O_j[j]:
            op = int(op)
            model.addConstr(C_weld[j] >= C[op], name=f"Cweld_ge_{j}_{op}")

    # final completion
    for j in [int(v) for v in J]:
        model.addConstr(
            C_final[j] == C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j],
            name=f"Cfinal_{j}"
        )

    for j in [int(v) for v in J]:
        model.addConstr(T[j] >= C_final[j] - float(d_j[j]), name=f"Tdef_{j}")
        model.addConstr(C_max >= C_final[j], name=f"Cmax_ge_{j}")
        model.addConstr(T_max >= T[j], name=f"Tmax_ge_{j}")

    if t0 is not None and free_ops:
        for i in [int(v) for v in free_ops]:
            if i in I_list:
                model.addConstr(S[i] >= float(t0), name=f"free_after_t0_{i}")

    # freeze DONE ops only
    if old_solution is not None:
        S_old = old_solution["S_old"]
        C_old = old_solution["C_old"]
        x_old = old_solution["x_old"]
        y_old = old_solution["y_old"]

        done_set = set(int(v) for v in freeze_done_ops)

        for i in done_set:
            if i in I_list:
                model.addConstr(S[i] == float(S_old[i]), name=f"freeze_done_S_{i}")
                model.addConstr(C[i] == float(C_old[i]), name=f"freeze_done_C_{i}")

        for (i, m) in list(x.keys()):
            if int(i) in done_set:
                model.addConstr(x[i, m] == int(x_old.get((int(i), int(m)), 0)), name=f"freeze_done_x_{i}_{m}")
        for (i, l) in list(y.keys()):
            if int(i) in done_set:
                model.addConstr(y[i, l] == int(y_old.get((int(i), int(l)), 0)), name=f"freeze_done_y_{i}_{l}")

    model.update()
    return model, x, y, S, C, C_weld, C_final, T, T_max, C_max, keep


# ==========================================================
#  SOLVE WRAPPERS
# ==========================================================

def solve_baseline(data, plan_start_iso: str, method: str = "mip"):
    """
    method:
      - "mip"       -> current Gurobi model
      - "heuristic" -> ATCS list scheduling
    """
    method = (method or "mip").lower().strip()

    if method == "heuristic":
        sched, obj = heuristic_schedule(
            data,
            fixed_ops=None,
            start_time_floor=0.0,
            k1=2.0,
            unavailable_machines=None,
            unavailable_stations=None
        )

        S_old = {int(r["op_id"]): float(r["start"]) for r in sched}
        C_old = {int(r["op_id"]): float(r["finish"]) for r in sched}

        x_old = {}
        y_old = {}
        for r in sched:
            i = int(r["op_id"])
            m = r.get("machine", None)
            l = r.get("station", None)
            if m is not None:
                x_old[_tuplekey_to_str(i, int(m))] = 1
            if l is not None:
                y_old[_tuplekey_to_str(i, int(l))] = 1

        return {
            "plan_start_iso": plan_start_iso,
            "objective": {"T_max": float(obj["T_max"]), "C_max": float(obj["C_max"])},
            "schedule": sched,
            "S_old": S_old, "C_old": C_old,
            "x_old": x_old, "y_old": y_old
        }

    model, x, y, S, C, Cw, Cf, T, Tmax, Cmax, _ = build_model("Baseline", data)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"No feasible baseline. Status={model.Status}")

    I = data["I"]
    S_old, C_old, x_old, y_old = extract_old_solution(I, x, y, S, C)

    job_of = op_to_job_map(data["O_j"])
    chosen_m, chosen_l = extract_assignment_for_ops(I, data["M_i"], data["L_i"], x, y)

    schedule = []
    for i in I:
        i = int(i)
        schedule.append({
            "op_id": i,
            "op_label": str(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(S[i].X),
            "finish": float(C[i].X),
            "machine": chosen_m[i],
            "station": chosen_l[i],
        })

    return {
        "plan_start_iso": plan_start_iso,
        "objective": {"T_max": float(Tmax.X), "C_max": float(Cmax.X)},
        "schedule": schedule,
        "S_old": S_old, "C_old": C_old,
        "x_old": x_old, "y_old": y_old
    }


def solve_reschedule(
    data_base,
    old_solution,
    urgent_payload: dict,
    unavailable_machines=None,
    unavailable_stations=None,
    mode="continue",
    method: str = "mip",
):
    """
    method:
      - "mip"       -> current Gurobi rescheduling model
      - "heuristic" -> fixes done/running (baseline), schedules remainder by ATCS heuristic

    IMPORTANT RULE:
      - If there is a machine/station breakdown (unavailable_* not empty),
        a running operation on that resource CANNOT "continue & finish" there.
        => It must be cut at t0 and remainder must be assigned elsewhere.
      - For urgent job / due date changes WITHOUT breakdown, model may keep or preempt.
    """
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []
    method = (method or "mip").lower().strip()

    old_solution = normalize_old_solution(old_solution)

    if "plan_start_iso" not in old_solution:
        raise ValueError("baseline_solution.json is missing plan_start_iso. Re-run baseline.")

    plan_calendar = old_solution.get("plan_calendar", {})
    t0 = compute_t0_from_plan_start(old_solution["plan_start_iso"], plan_calendar=plan_calendar)

    I0 = data_base["I"]
    I_done, I_run, I_free = classify_ops_by_t0(I0, old_solution["S_old"], old_solution["C_old"], t0)

    data1 = add_urgent_job_from_payload(data_base, t0=t0, urgent_payload=urgent_payload)
    data2 = add_split_remainders_for_running_ops(data1, I_run, t0, old_solution)

    if method == "heuristic":
        job_of = op_to_job_map(data2["O_j"])

        x_old = old_solution.get("x_old", {})
        y_old = old_solution.get("y_old", {})

        badM = set(int(m) for m in unavailable_machines)
        badL = set(int(l) for l in unavailable_stations)

        def old_machine(i):
            for (ii, m), v in x_old.items():
                if int(ii) == int(i) and int(v) == 1:
                    return int(m)
            return None

        def old_station(i):
            for (ii, l), v in y_old.items():
                if int(ii) == int(i) and int(v) == 1:
                    return int(l)
            return None

        fixed = {}

        for i in I_done:
            i = int(i)
            fixed[i] = {
                "op_id": i,
                "op_label": str(i),
                "job_id": int(job_of.get(i, -1)),
                "start": float(old_solution["S_old"][i]),
                "finish": float(old_solution["C_old"][i]),
                "machine": old_machine(i),
                "station": old_station(i),
            }

        for i in I_run:
            i = int(i)
            m0 = old_machine(i)
            l0 = old_station(i)

            start_i = float(old_solution["S_old"][i])
            finish_i = float(old_solution["C_old"][i])

            if (m0 is not None and m0 in badM) or (l0 is not None and l0 in badL):
                finish_i = float(t0)

            fixed[i] = {
                "op_id": i,
                "op_label": str(i),
                "job_id": int(job_of.get(i, -1)),
                "start": float(start_i),
                "finish": float(finish_i),
                "machine": m0,
                "station": l0,
            }

        sched, obj = heuristic_schedule(
            data2,
            fixed_ops=fixed,
            start_time_floor=float(t0),
            k1=2.0,
            unavailable_machines=unavailable_machines,
            unavailable_stations=unavailable_stations,
        )

        urgent = int(data2.get("urgent_job_id"))
        urgent_ops = data2.get("urgent_ops", [])

        C_by_op = {int(r["op_id"]): float(r["finish"]) for r in sched}
        Cw_u = 0.0
        for op in data2["O_j"][urgent]:
            Cw_u = max(Cw_u, C_by_op[int(op)])
        Cf_u = (
            Cw_u
            + data2["g_j"][urgent] * data2["t_grind_j"][urgent]
            + data2["p_j"][urgent] * data2["t_paint_j"][urgent]
        )
        T_u = max(0.0, Cf_u - float(data2["d_j"][urgent]))

        return {
            "t0": float(t0),
            "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free},
            "objective": {"T_max": float(obj["T_max"]), "C_max": float(obj["C_max"])},
            "urgent": {
                "job_id": urgent,
                "ops": [int(i) for i in urgent_ops],
                "C_final": float(Cf_u),
                "T": float(T_u),
                "d": float(data2["d_j"][urgent])
            },
            "keep_decisions": {},
            "schedule": sched,
            "changed_ops": []
        }

    model, x, y, S, C, Cw, Cf, T, Tmax, Cmax, keep = build_model(
        name="Reschedule",
        data=data2,
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        freeze_done_ops=I_done,
        running_ops=I_run,
        mode=mode,
        t0=t0,
        free_ops=I_free,
        old_solution=old_solution
    )
    model.optimize()

    if model.SolCount == 0:
        model.computeIIS()
        model.write("iis.ilp")
        raise RuntimeError(f"No feasible reschedule. Status={model.Status}")

    urgent = int(data2.get("urgent_job_id"))
    urgent_ops = data2.get("urgent_ops", [])

    keep_out = {}
    if mode == "optimize" and keep is not None:
        for i in I_run:
            keep_out[int(i)] = int(round(keep[int(i)].X))

    job_of = op_to_job_map(data2["O_j"])
    chosen_m, chosen_l = extract_assignment_for_ops(data2["I"], data2["M_i"], data2["L_i"], x, y)

    rem_map = data2.get("rem_map", {})
    rem_to_orig = {int(v): int(k) for k, v in rem_map.items()}

    schedule_new = []
    for i in data2["I"]:
        i = int(i)
        label = str(i)
        if i in rem_to_orig:
            label = f"{rem_to_orig[i]}(cont)"
        schedule_new.append({
            "op_id": i,
            "op_label": label,
            "job_id": int(job_of.get(i, -1)),
            "start": float(S[i].X),
            "finish": float(C[i].X),
            "machine": chosen_m[i],
            "station": chosen_l[i],
        })

    changed_ops = []
    for i in data_base["I"]:
        old_s = old_solution["S_old"].get(int(i), None)
        old_c = old_solution["C_old"].get(int(i), None)
        if old_s is None:
            continue

        new_s = float(S[int(i)].X) if int(i) in S else None
        new_c = float(C[int(i)].X) if int(i) in C else None
        if new_s is None or new_c is None:
            continue

        if abs(new_s - old_s) > 1e-6 or abs(new_c - old_c) > 1e-6:
            changed_ops.append({
                "op_id": int(i),
                "old_start": float(old_s), "old_finish": float(old_c),
                "new_start": float(new_s), "new_finish": float(new_c),
            })

    return {
        "t0": float(t0),
        "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free},
        "objective": {"T_max": float(Tmax.X), "C_max": float(Cmax.X)},
        "urgent": {
            "job_id": urgent,
            "ops": [int(i) for i in urgent_ops],
            "C_final": float(Cf[urgent].X),
            "T": float(T[urgent].X),
            "d": float(data2["d_j"][urgent])
        },
        "keep_decisions": keep_out,
        "schedule": schedule_new,
        "changed_ops": changed_ops
    }
