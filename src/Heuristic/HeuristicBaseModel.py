# -*- coding: utf-8 -*-
"""
HeuristicBaseModel.py

Welding Shop Scheduling – Giffler–Thompson (GT) + ATC Heuristic (BASE)

DATA-AGNOSTIC: Provide a `data` dictionary (see scenarios) and call:
    res = run_heuristic(data, k1=2.0)

REPORT-ALIGNED CORE (kept as close as possible):
- GT pivot selection: i* = argmin earliest completion among STARTABLE ops at time t
- Conflict set: shares pivot's chosen machine OR station
- ATC selection within conflict set
- Tardiness: T_j = max(C_final_j - d_j, 0)

CRITICAL FIX (still GT-consistent, but enables parallelism):
- "Batch scheduling at fixed t": while there exist operations that can START at time t,
  schedule them one-by-one (GT pivot + ATC) WITHOUT advancing t.
- Only when nothing can start at time t, advance t to the next event (min of machine/station availability
  or next release/precedence-ready time).

This fixes the observed issue where machine 2 stays unused and makespan doubles
because the heuristic behaves like a single-thread pipeline.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


# -------------------------------
# Utilities
# -------------------------------

def get_job_of_map(J: List[int], O_j: Dict[int, List[int]]) -> Dict[int, int]:
    job_of: Dict[int, int] = {}
    for j in J:
        for i in O_j[j]:
            job_of[i] = j
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

def run_heuristic(data: Dict[str, Any], k1: float = 2.0, eps: float = 1e-9) -> HeuristicResult:
    """
    Report-aligned GT + ATC heuristic with a minimal but necessary fix:
    fill all operations that can start at the current decision time t (batching),
    then advance t to the next event.
    """
    verify_data_basic(data)

    J = data["J"]
    I = data["I"]
    O_j = data["O_j"]
    M = data["M"]
    L = data["L"]
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
    L_small = data["L_small"]

    job_of = get_job_of_map(J, O_j)

    # Average processing time for ATC scaling
    all_p = [float(p_im[(i, m)]) for i in I for m in M_i[i]]
    p_bar = sum(all_p) / len(all_p) if all_p else 1.0

    # Resource availability times
    avail_machine: Dict[int, float] = {m: 0.0 for m in M}
    avail_station: Dict[int, float] = {l: 0.0 for l in L}

    # Solution containers
    S: Dict[int, float] = {}
    C: Dict[int, float] = {}
    assign_machine: Dict[int, int] = {}
    assign_station: Dict[int, int] = {}

    scheduled = set()
    t = 0.0  # decision time

    def atc_index(i: int, t_now: float) -> float:
        j = job_of[i]
        p_i = min(float(p_im[(i, m)]) for m in M_i[i])
        slack = max(float(d_j[j]) - p_i - t_now, 0.0)
        denom = max(k1 * p_bar, eps)
        return (1.0 / max(p_i, eps)) * math.exp(-slack / denom)

    def ready_time(i: int) -> float:
        """R_i = max(release(job), completion(predecessors))."""
        j = job_of[i]
        rt = float(r_j[j])
        if Pred_i[i]:
            rt = max(rt, max(C[h] for h in Pred_i[i]))
        return rt

    while len(scheduled) < len(I):

        # ---- Build eligible candidates + their ready times
        candidates: List[int] = []
        R_i: Dict[int, float] = {}

        for i in I:
            if i in scheduled:
                continue
            if any(h not in scheduled for h in Pred_i[i]):
                continue
            R_i[i] = ready_time(i)
            candidates.append(i)

        if not candidates:
            raise RuntimeError("No eligible candidates (precedence cycle or inconsistent data).")

        # ---- If nothing is released by time t, move t to earliest ready time (report-aligned jump)
        if all(R_i[i] > t + eps for i in candidates):
            t = min(R_i[i] for i in candidates)

        # ==========================================================
        # BATCH SCHEDULING AT FIXED t (parallelism fix)
        # ==========================================================
        while True:
            # Ready at current t
            ready = [i for i in candidates if R_i[i] <= t + eps]
            if not ready:
                break

            best_pair: Dict[int, Tuple[int, int]] = {}
            earliest_start: Dict[int, float] = {}
            earliest_completion: Dict[int, float] = {}

            # For each ready op, compute best (m,l) under current availabilities
            for i in ready:
                bestS = float("inf")
                bestC = float("inf")
                bestML: Tuple[int, int] | None = None

                machines_sorted = sorted(M_i[i], key=lambda mm: (avail_machine[mm], mm))
                stations_sorted = sorted(L_i[i], key=lambda ll: (avail_station[ll], ll))

                for m in machines_sorted:
                    for l in stations_sorted:
                        if beta_i[i] == 1 and l in L_small:
                            continue

                        s = max(t, R_i[i], avail_machine[m], avail_station[l])
                        c = s + float(p_im[(i, m)])

                        # Primary: minimize start time (so we can fill resources at t)
                        # Secondary: minimize completion, then deterministic tie-break on ids
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

            # Startable now: earliest start is exactly t
            startable = [i for i in ready if i in earliest_start and abs(earliest_start[i] - t) <= eps]
            if not startable:
                break

            # Pivot i* among STARTABLE: min earliest completion (GT pivot)
            i_star = min(startable, key=lambda ii: earliest_completion[ii])
            m_star, l_star = best_pair[i_star]

            # Conflict set among STARTABLE: shares pivot machine OR station
            conflict_set = [i_star]
            for i in startable:
                if i == i_star:
                    continue
                mi, li = best_pair[i]
                if (mi == m_star) or (li == l_star):
                    conflict_set.append(i)

            # Choose within conflict set by ATC
            chosen = max(conflict_set, key=lambda ii: atc_index(ii, t))

            m_chosen, l_chosen = best_pair[chosen]

            # By construction, chosen is startable => start == t
            start = t
            comp = start + float(p_im[(chosen, m_chosen)])

            # Commit decision
            S[chosen] = start
            C[chosen] = comp
            assign_machine[chosen] = m_chosen
            assign_station[chosen] = l_chosen

            avail_machine[m_chosen] = comp
            avail_station[l_chosen] = comp
            scheduled.add(chosen)

            # Remove from candidate pool and continue batching at SAME t
            candidates.remove(chosen)
            if not candidates:
                break

            # Update ready times for newly eligible ops (if any became eligible at same t)
            # (Predecessor completion is at >= t, so new ops may become ready only at >t, but safe to refresh.)
            for i2 in candidates:
                # still eligible by precedence (they are in candidates), just refresh R_i in case of multiple preds
                R_i[i2] = ready_time(i2)

        # ==========================================================
        # Advance t to next event ONLY when no more startable at t
        # ==========================================================
        if len(scheduled) < len(I):
            next_t = float("inf")

            # next machine/station completion strictly after t
            for tt in avail_machine.values():
                if tt > t + eps:
                    next_t = min(next_t, tt)
            for tt in avail_station.values():
                if tt > t + eps:
                    next_t = min(next_t, tt)

            # next ready/release among remaining eligible candidates
            # (we must recompute candidates here because batching may have emptied candidates)
            remaining_candidates = []
            R_rem: List[float] = []
            for i in I:
                if i in scheduled:
                    continue
                if any(h not in scheduled for h in Pred_i[i]):
                    continue
                rt = ready_time(i)
                remaining_candidates.append(i)
                R_rem.append(rt)

            for rt in R_rem:
                if rt > t + eps:
                    next_t = min(next_t, rt)

            if next_t == float("inf"):
                raise RuntimeError("Stuck: cannot advance time (check feasibility or eps).")

            t = next_t

    # ---- Job metrics
    C_weld: Dict[int, float] = {}
    C_final: Dict[int, float] = {}
    T: Dict[int, float] = {}

    for j in J:
        last_op = O_j[j][-1]
        C_weld[j] = C[last_op]
        C_final[j] = (
            C_weld[j]
            + float(g_j[j]) * float(t_grind_j[j])
            + float(p_flag_j[j]) * float(t_paint_j[j])
        )
        T[j] = max(C_final[j] - float(d_j[j]), 0.0)

    C_max = max(C_final.values())
    T_max = max(T.values())

    return HeuristicResult(
        S=S, C=C,
        assign_machine=assign_machine,
        assign_station=assign_station,
        C_weld=C_weld,
        C_final=C_final,
        T=T,
        T_max=T_max,
        C_max=C_max
    )


# -------------------------------
# Feasibility checker
# -------------------------------

def check_heuristic_solution(data: Dict[str, Any], res: HeuristicResult, tol: float = 1e-6) -> None:
    J = data["J"]
    I = data["I"]
    O_j = data["O_j"]
    M = data["M"]
    L = data["L"]
    Pred_i = data["Pred_i"]
    r_j = data["r_j"]
    beta_i = data["beta_i"]
    L_small = data["L_small"]

    S, C = res.S, res.C
    am, al = res.assign_machine, res.assign_station
    job_of = get_job_of_map(J, O_j)

    # precedence
    for i in I:
        for h in Pred_i[i]:
            if S[i] + tol < C[h]:
                raise AssertionError(
                    f"Precedence violated: op {i} starts {S[i]:.3f} < C[{h}]={C[h]:.3f}"
                )

    # release
    for i in I:
        j = job_of[i]
        if S[i] + tol < float(r_j[j]):
            raise AssertionError(
                f"Release violated: op {i} starts {S[i]:.3f} < r_j[{j}]={r_j[j]:.3f}"
            )

    # machine non-overlap
    for m in M:
        ops_m = [i for i in I if am.get(i) == m]
        for a in range(len(ops_m)):
            for b in range(a + 1, len(ops_m)):
                i, h = ops_m[a], ops_m[b]
                if not (C[i] <= S[h] + tol or C[h] <= S[i] + tol):
                    raise AssertionError(f"Machine {m} overlap: ops {i} and {h}")

    # station non-overlap
    for l in L:
        ops_l = [i for i in I if al.get(i) == l]
        for a in range(len(ops_l)):
            for b in range(a + 1, len(ops_l)):
                i, h = ops_l[a], ops_l[b]
                if not (C[i] <= S[h] + tol or C[h] <= S[i] + tol):
                    raise AssertionError(f"Station {l} overlap: ops {i} and {h}")

    # big-station constraint
    for i in I:
        if beta_i[i] == 1 and al[i] in L_small:
            raise AssertionError(
                f"Big-station violated: op {i} assigned to small station {al[i]}"
            )

    print("All heuristic feasibility checks passed.")
