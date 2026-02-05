# -*- coding: utf-8 -*-
"""
HeuristicBaseModel.py

Welding Shop Scheduling – Giffler–Thompson (GT) + ATC Heuristic (BASE + RESCHED SUPPORT)

ADD-ON (Urgent priority):
- If data contains "urgent_job_id" (int), ATC priority is multiplied by "urgent_weight" (default 25.0)
  for operations of that urgent job. This makes urgent ops preferred whenever they are startable.

RESCHED SUPPORT:
- fixed_ops: pre-scheduled ops that must remain at given start/finish and block their resources until finish
- start_time_floor: no non-fixed op can start before this (use t0)
- unavailable resources: prevent new assignments to those machines/stations

Still report-aligned:
- GT pivot: among STARTABLE at time t, pick min earliest completion
- Conflict: shares pivot machine OR station
- ATC within conflict
- Batch scheduling at fixed t (parallelism fix)

Python 3.9 compatible.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


# -------------------------------
# Utilities
# -------------------------------

def get_job_of_map(J: List[int], O_j: Dict[int, List[int]]) -> Dict[int, int]:
    job_of: Dict[int, int] = {}
    for j in J:
        for i in O_j[j]:
            job_of[int(i)] = int(j)
    return job_of


def verify_data_basic(data: Dict[str, Any]) -> None:
    """
    Lightweight checks: feasibility sets non-empty, precedence references valid, p_im exists.
    """
    required = [
        "J", "I", "O_j", "M", "L", "M_i", "L_i", "Pred_i", "p_im", "r_j", "d_j",
        "g_j", "p_flag_j", "t_grind_j", "t_paint_j", "beta_i", "L_small"
    ]
    missing_keys = [k for k in required if k not in data]
    if missing_keys:
        raise ValueError(f"Data dict missing keys: {missing_keys}")

    J = data["J"]
    I = data["I"]
    O_j = data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    r_j = data["r_j"]
    d_j = data["d_j"]

    all_ops = set()
    for j in J:
        for op in O_j[j]:
            op = int(op)
            all_ops.add(op)
            if op not in I:
                raise ValueError(f"Operation {op} in O_j[{j}] not in I")

    extra = set(I) - all_ops
    if extra:
        raise ValueError(f"Operations in I but not in any job: {sorted(extra)}")

    bad_m = [i for i in I if i not in M_i or len(M_i[i]) == 0]
    bad_l = [i for i in I if i not in L_i or len(L_i[i]) == 0]
    if bad_m:
        raise ValueError(f"Ops with no feasible machines: {bad_m}")
    if bad_l:
        raise ValueError(f"Ops with no feasible stations: {bad_l}")

    bad_pred = [(i, h) for i in I for h in Pred_i.get(i, []) if h not in I]
    if bad_pred:
        raise ValueError(f"Invalid Pred_i references (sample): {bad_pred[:10]}")

    missing_p = [(i, m) for i in I for m in M_i[i] if (i, m) not in p_im]
    if missing_p:
        raise ValueError(f"Missing p_im entries (sample): {missing_p[:10]}")

    if set(r_j.keys()) != set(J):
        raise ValueError("r_j not defined for all jobs")
    if set(d_j.keys()) != set(J):
        raise ValueError("d_j not defined for all jobs")


@dataclass
class HeuristicResult:
    S: Dict[int, float]
    C: Dict[int, float]
    assign_machine: Dict[int, int]
    assign_station: Dict[int, int]
    C_weld: Dict[int, float]
    C_final: Dict[int, float]
    T: Dict[int, float]
    T_max: float
    C_max: float


# -------------------------------
# GT + ATC heuristic (Report-aligned + parallel batching at fixed t)
# -------------------------------

def run_heuristic(
    data: Dict[str, Any],
    k1: float = 2.0,
    eps: float = 1e-9,
    fixed_ops: Optional[Dict[int, Dict[str, Any]]] = None,
    start_time_floor: float = 0.0,
    unavailable_machines: Optional[List[int]] = None,
    unavailable_stations: Optional[List[int]] = None,
) -> HeuristicResult:
    """
    GT + ATC heuristic with rescheduling support.

    fixed_ops format (op_id -> dict):
      {
        "start": float,
        "finish": float,
        "machine": int or None,
        "station": int or None
      }

    Rules:
    - Fixed ops are taken as given and block their resources until finish.
    - Non-fixed ops cannot start before start_time_floor (use t0).
    - New assignments cannot use unavailable machines/stations.
    """
    verify_data_basic(data)

    fixed_ops = fixed_ops or {}
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []

    badM = set(int(x) for x in unavailable_machines)
    badL = set(int(x) for x in unavailable_stations)

    J = data["J"]
    I = [int(x) for x in data["I"]]
    O_j = data["O_j"]
    M = [int(x) for x in data["M"]]
    L = [int(x) for x in data["L"]]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    r_j = data["r_j"]
    d_j = data["d_j"]
    g_j = data["g_j"]
    p_flag_j = data["p_flag_j"]
    t_grind_j = data["t_grind_j"]
    t_paint_j = data["t_paint_j"]
    p_im = data["p_im"]
    beta_i = data["beta_i"]
    L_small = set(int(x) for x in data["L_small"])

    job_of = get_job_of_map(J, O_j)

    # Urgent boosting
    urgent_job_id = data.get("urgent_job_id", None)
    try:
        urgent_job_id = int(urgent_job_id) if urgent_job_id is not None else None
    except Exception:
        urgent_job_id = None
    urgent_weight = float(data.get("urgent_weight", 25.0))

    # Average processing time for ATC scaling
    all_p = [float(p_im[(int(i), int(m))]) for i in I for m in M_i[i]]
    p_bar = (sum(all_p) / len(all_p)) if all_p else 1.0

    # Resource availability times
    avail_machine: Dict[int, float] = {m: 0.0 for m in M}
    avail_station: Dict[int, float] = {l: 0.0 for l in L}

    # Solution containers
    S: Dict[int, float] = {}
    C: Dict[int, float] = {}
    assign_machine: Dict[int, int] = {}
    assign_station: Dict[int, int] = {}

    scheduled = set()

    # ------------------------------------------------
    # Load FIXED OPS
    # ------------------------------------------------
    for op_id, row in fixed_ops.items():
        i = int(op_id)
        if i not in I:
            continue
        s = float(row["start"])
        c = float(row["finish"])
        S[i] = s
        C[i] = c
        scheduled.add(i)

        m = row.get("machine", None)
        l = row.get("station", None)

        if m is not None:
            m = int(m)
            assign_machine[i] = m
            avail_machine[m] = max(avail_machine.get(m, 0.0), c)
        if l is not None:
            l = int(l)
            assign_station[i] = l
            avail_station[l] = max(avail_station.get(l, 0.0), c)

    # decision time starts at floor (t0 for reschedule)
    t = float(start_time_floor)

    def ready_time(i: int) -> float:
        """R_i = max(release(job), completion(predecessors))."""
        j = job_of[i]
        rt = float(r_j[j])

        preds = Pred_i.get(i, [])
        if preds:
            rt = max(rt, max(C[int(h)] for h in preds))

        # enforce floor for non-fixed ops
        rt = max(rt, float(start_time_floor))
        return rt

    def atc_index(i: int, t_now: float, best_pair_local: Dict[int, Tuple[int, int]]) -> float:
        j = job_of[i]
        m_earliest, _ = best_pair_local[i]
        p_i = float(p_im[(int(i), int(m_earliest))])
        slack = max(float(d_j[j]) - p_i - t_now, 0.0)
        denom = max(k1 * p_bar, eps)
        base = (1.0 / max(p_i, eps)) * math.exp(-slack / denom)

        # Boost urgent job ops (only if configured)
        if urgent_job_id is not None and int(j) == int(urgent_job_id):
            base *= max(1.0, urgent_weight)

        return base

    def preds_done(i: int) -> bool:
        return all(int(h) in C for h in Pred_i.get(i, []))

    # ------------------------------------------------
    # Main loop
    # ------------------------------------------------
    while len(scheduled) < len(I):

        # Build eligible candidates (precedence-feasible) that are not fixed/scheduled
        candidates: List[int] = []
        R_i: Dict[int, float] = {}

        for i in I:
            if i in scheduled:
                continue
            if not preds_done(i):
                continue
            R_i[i] = ready_time(i)
            candidates.append(i)

        if not candidates:
            raise RuntimeError("No eligible candidates (precedence cycle or inconsistent data).")

        # If none released by current t, jump t to min ready time
        if all(R_i[i] > t + eps for i in candidates):
            t = min(R_i[i] for i in candidates)

        # ==========================================================
        # Batch scheduling at fixed t
        # ==========================================================
        while True:
            ready = [i for i in candidates if R_i[i] <= t + eps]
            if not ready:
                break

            best_pair: Dict[int, Tuple[int, int]] = {}
            earliest_start: Dict[int, float] = {}
            earliest_completion: Dict[int, float] = {}

            for i in ready:
                bestS = float("inf")
                bestC = float("inf")
                bestML = None  # type: Optional[Tuple[int, int]]

                machines_sorted = sorted(
                    [int(mm) for mm in M_i[i] if int(mm) not in badM],
                    key=lambda mm: (avail_machine.get(mm, 0.0), mm)
                )
                stations_sorted = sorted(
                    [int(ll) for ll in L_i[i] if int(ll) not in badL],
                    key=lambda ll: (avail_station.get(ll, 0.0), ll)
                )

                if not machines_sorted or not stations_sorted:
                    continue

                for m in machines_sorted:
                    for l in stations_sorted:
                        if int(beta_i.get(i, 0)) == 1 and l in L_small:
                            continue

                        s = max(t, R_i[i], avail_machine[m], avail_station[l])
                        c = s + float(p_im[(int(i), int(m))])

                        # Primary: minimize start time; Secondary: minimize completion
                        if (s < bestS - 1e-12) or (
                            abs(s - bestS) <= 1e-12 and (c < bestC - 1e-12)
                        ) or (
                            abs(s - bestS) <= 1e-12 and abs(c - bestC) <= 1e-12 and bestML is not None and (m, l) < bestML
                        ):
                            bestS = s
                            bestC = c
                            bestML = (m, l)

                if bestML is None:
                    continue

                best_pair[i] = bestML
                earliest_start[i] = bestS
                earliest_completion[i] = bestC

            # startable now: earliest start == t
            startable = [i for i in ready if i in earliest_start and abs(earliest_start[i] - t) <= eps]
            if not startable:
                break

            # GT pivot among STARTABLE: min earliest completion
            i_star = min(startable, key=lambda ii: earliest_completion[ii])
            m_star, l_star = best_pair[i_star]

            # conflict set among STARTABLE: shares pivot machine OR station
            conflict_set = [i_star]
            for i in startable:
                if i == i_star:
                    continue
                mi, li = best_pair[i]
                if (mi == m_star) or (li == l_star):
                    conflict_set.append(i)

            chosen = max(conflict_set, key=lambda ii: atc_index(ii, t, best_pair))
            m_chosen, l_chosen = best_pair[chosen]

            # start = t (by construction)
            start = t
            comp = start + float(p_im[(int(chosen), int(m_chosen))])

            S[chosen] = float(start)
            C[chosen] = float(comp)
            assign_machine[chosen] = int(m_chosen)
            assign_station[chosen] = int(l_chosen)

            avail_machine[int(m_chosen)] = float(comp)
            avail_station[int(l_chosen)] = float(comp)

            scheduled.add(chosen)
            candidates.remove(chosen)

            if not candidates:
                break

            # refresh R_i for remaining candidates
            for i2 in candidates:
                R_i[i2] = ready_time(i2)

        # ==========================================================
        # Advance t to next event (when nothing can start at t)
        # ==========================================================
        if len(scheduled) < len(I):
            next_t = float("inf")

            for tt in avail_machine.values():
                if tt > t + eps:
                    next_t = min(next_t, tt)
            for tt in avail_station.values():
                if tt > t + eps:
                    next_t = min(next_t, tt)

            # next ready time among precedence-feasible but not yet released
            for i in I:
                if i in scheduled:
                    continue
                if not preds_done(i):
                    continue
                rt = ready_time(i)
                if rt > t + eps:
                    next_t = min(next_t, rt)

            if next_t == float("inf"):
                raise RuntimeError("Stuck: cannot advance time (check feasibility or eps).")

            t = next_t

    # ------------------------------------------------
    # Job metrics (ROBUST: C_weld = max over job ops)
    # ------------------------------------------------
    C_weld: Dict[int, float] = {}
    C_final: Dict[int, float] = {}
    T: Dict[int, float] = {}

    for j in J:
        cwj = 0.0
        for op in O_j[j]:
            cwj = max(cwj, float(C[int(op)]))
        C_weld[int(j)] = float(cwj)

        cf = (
            float(cwj)
            + float(g_j[int(j)]) * float(t_grind_j[int(j)])
            + float(p_flag_j[int(j)]) * float(t_paint_j[int(j)])
        )
        C_final[int(j)] = float(cf)
        T[int(j)] = max(float(cf) - float(d_j[int(j)]), 0.0)

    C_max = max(C_final.values()) if C_final else 0.0
    T_max = max(T.values()) if T else 0.0

    return HeuristicResult(
        S=S, C=C,
        assign_machine=assign_machine,
        assign_station=assign_station,
        C_weld=C_weld,
        C_final=C_final,
        T=T,
        T_max=float(T_max),
        C_max=float(C_max)
    )
