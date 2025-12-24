#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 12:03:12 2025

@author: ezgieker
"""

#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Welding Shop Scheduling – HEURISTIC (GT + ATCS) using the SAME DATA
- 8 jobs, 30 ops
- 12 machines (TIG/MAG), 4 stations (only station 1 is big)
- Most operations require BIG station
- Data arranged to create tardiness

This script DOES NOT solve MIP with Gurobi.
It builds a feasible schedule using:
  Giffler–Thompson (GT) selection + ATCS-style priority rule
and then checks feasibility + prints results + plots Gantt charts.

Author: Teknopar / Heuristic version requested
"""

import math
from collections import defaultdict

# Gurobi is imported because you asked "gurobi kodu",
# but here we don't optimize; we only use it optionally later.
from gurobipy import Model, GRB  # noqa: F401

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ===============================
#  DATA (8 jobs, 30 ops)  (same as you sent)
# ===============================

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

M = list(range(1, 13))   # 1..12 machines
L = [1, 2, 3, 4]         # 4 stations
L_big = [1]
L_small = [2, 3, 4]

# Machines types
# 1..3 TIG, 4..12 MAG
machine_type = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2
}

# Operation -> allowed machine TYPES (odd TIG, even MAG)
K_i = {}
for i in I:
    K_i[i] = [1] if (i % 2 == 1) else [2]

# Operation -> feasible machines
M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

# Stations feasible (all)
L_i = {i: [1, 2, 3, 4] for i in I}

# Precedence
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

# Job 8 no internal precedence (except the "last after all" rule below)

# Extra step: last op after all ops of same job
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for op in ops:
        if op != last:
            preds.add(op)
    Pred_i[last] = list(preds)

# Processing times p_im (HEAVIER)
p_im = {}
for i in I:
    for m in M_i[i]:
        base = 6 + (i % 7)            # 6..12
        machine_add = (m - 1) * 0.8   # spread
        p_im[(i, m)] = float(base + machine_add)

# Release times r_j (0,2,4,...,14)
r_j = {j: float(2 * (j - 1)) for j in J}

# Due dates d_j (tight)
# 22..29
Iend = [O_j[j][-1] for j in J]
d_i = {}
due_base = 22.0
for idx, i_last in enumerate(Iend):
    d_i[i_last] = due_base + 1.0 * idx

d_j = {j: d_i[O_j[j][-1]] for j in J}

# Grinding requirement g_j, Painting requirement p_j
g_j = {}
p_j = {}
for idx, j in enumerate(J):
    g_j[j] = 1 if (idx % 2 == 0) else 0
    p_j[j] = 1 if (idx in [0, 1, 2]) else 0

# grinding/painting times
t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

# Big-station requirement beta_i
# Only FIRST operation flexible; others MUST be big (station 1)
beta_i = {}
first_ops = [O_j[j][0] for j in J]
for i in I:
    beta_i[i] = 0 if (i in first_ops) else 1


# ===============================
#  HELPERS
# ===============================

def op_job_map(J, O_j):
    mp = {}
    for j in J:
        for op in O_j[j]:
            mp[op] = j
    return mp


JOB_OF = op_job_map(J, O_j)


def atcs_index(i, t_now, pbar, k1=2.0):
    """
    ATCS-like index (simplified):
      ATCS(i,t) = (1/pi) * exp( - max(d_job - pi - t, 0) / (k1*pbar) )
    where pi = min_m p_im(i,m)
    """
    j = JOB_OF[i]
    pi = min(p_im[(i, m)] for m in M_i[i])
    slack = max(d_j[j] - pi - t_now, 0.0)
    return (1.0 / max(pi, 1e-9)) * math.exp(-slack / max(k1 * pbar, 1e-9))


def compute_pbar():
    vals = []
    for i in I:
        for m in M_i[i]:
            vals.append(p_im[(i, m)])
    return sum(vals) / max(len(vals), 1)


def check_no_overlap(intervals, tol=1e-9):
    """
    intervals: list of (start, end, op)
    returns list of conflicts (op_a, op_b)
    """
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    conflicts = []
    for k in range(len(intervals) - 1):
        s1, e1, o1 = intervals[k]
        s2, e2, o2 = intervals[k + 1]
        if s2 < e1 - tol:
            conflicts.append((o1, o2))
    return conflicts


def check_heuristic_solution():
    """
    Checks:
    - precedence
    - release
    - machine overlap
    - station overlap
    - big station constraint
    """
    print("\n=== CHECK HEURISTIC SOLUTION ===")

    ok = True

    # precedence
    for i in I:
        for h in Pred_i[i]:
            if S[i] + 1e-9 < C[h]:
                ok = False
                print(f"[PREC] Violation: {h} -> {i} but S[{i}]={S[i]:.3f} < C[{h}]={C[h]:.3f}")

    # release: enforce for all ops (strong form)
    for i in I:
        j = JOB_OF[i]
        if S[i] + 1e-9 < r_j[j]:
            ok = False
            print(f"[REL] Violation: op {i} of job {j} starts before release "
                  f"S={S[i]:.3f} < r_j={r_j[j]:.3f}")

    # big station
    for i in I:
        if beta_i[i] == 1 and assign_station[i] not in L_big:
            ok = False
            print(f"[BIG] Violation: op {i} must be big but assigned station {assign_station[i]}")

    # machine overlaps
    for m in M:
        ints = [(S[i], C[i], i) for i in I if assign_machine[i] == m]
        conf = check_no_overlap(ints)
        if conf:
            ok = False
            for a, b in conf:
                print(f"[M-OVL] Machine {m} overlap between ops {a} and {b}")

    # station overlaps
    for l in L:
        ints = [(S[i], C[i], i) for i in I if assign_station[i] == l]
        conf = check_no_overlap(ints)
        if conf:
            ok = False
            for a, b in conf:
                print(f"[L-OVL] Station {l} overlap between ops {a} and {b}")

    if ok:
        print("OK: heuristic schedule is feasible.")
    else:
        print("ERROR: heuristic schedule has violations.")


# ===============================
#  HEURISTIC: GT + ATCS
# ===============================

def heuristic_schedule(k1=2.0):
    """
    Returns dictionaries:
      S[i], C[i], assign_machine[i], assign_station[i]
    GT+ATCS steps (simplified):
      - ready set by precedence + release
      - for each ready op i: best (m,l) minimizing completion given availability
      - choose i* with earliest completion (GT anchor)
      - conflict set: ops sharing same best m or best l with i*
      - select max ATCS from conflict set
      - fix operation, update resource availability, continue
    """

    # resource availability
    aM = {m: 0.0 for m in M}
    aL = {l: 0.0 for l in L}

    S_local = {}
    C_local = {}
    mu = {}
    lam = {}

    done = set()
    t = 0.0
    pbar = compute_pbar()

    # quick job release lookup
    def R_i(i):
        j = JOB_OF[i]
        pred_done_times = [C_local[h] for h in Pred_i[i]] if Pred_i[i] else []
        return max([r_j[j]] + pred_done_times) if pred_done_times else r_j[j]

    while len(done) < len(I):
        # candidate set: predecessors done
        Cand = [i for i in I if (i not in done) and all(h in done for h in Pred_i[i])]
        if not Cand:
            raise RuntimeError("No candidates found. Precedence cycle or inconsistent data?")

        # ready at time t
        ready = [i for i in Cand if R_i(i) <= t + 1e-9]
        if not ready:
            t = min(R_i(i) for i in Cand)
            ready = [i for i in Cand if R_i(i) <= t + 1e-9]

        # for each ready op, compute best (m,l) with earliest completion
        best_m = {}
        best_l = {}
        best_s = {}
        best_c = {}

        for i in ready:
            Ri = R_i(i)

            # feasible stations respecting big constraint
            feasible_stations = L_i[i]
            if beta_i[i] == 1:
                feasible_stations = [l for l in feasible_stations if l in L_big]

            best_ci = float("inf")
            best_pair = None
            best_si = None

            for m in M_i[i]:
                for l in feasible_stations:
                    si = max(t, Ri, aM[m], aL[l])
                    ci = si + p_im[(i, m)]
                    if ci < best_ci - 1e-12:
                        best_ci = ci
                        best_pair = (m, l)
                        best_si = si

            if best_pair is None:
                raise RuntimeError(f"No feasible (m,l) pair for op {i}. Check data/beta_i.")
            best_m[i], best_l[i] = best_pair
            best_s[i] = best_si
            best_c[i] = best_ci

        # GT anchor: i* = argmin earliest completion
        i_star = min(ready, key=lambda i: best_c[i])
        m_star = best_m[i_star]
        l_star = best_l[i_star]

        # conflict set: those who want same best machine OR same best station
        Kset = [i for i in ready if (best_m[i] == m_star) or (best_l[i] == l_star)]
        if i_star not in Kset:
            Kset.append(i_star)

        # ATCS selection among conflict set
        chosen = max(Kset, key=lambda i: atcs_index(i, t, pbar, k1=k1))

        # fix chosen op
        m_ch = best_m[chosen]
        l_ch = best_l[chosen]
        s_ch = best_s[chosen]
        c_ch = s_ch + p_im[(chosen, m_ch)]

        S_local[chosen] = s_ch
        C_local[chosen] = c_ch
        mu[chosen] = m_ch
        lam[chosen] = l_ch

        aM[m_ch] = c_ch
        aL[l_ch] = c_ch
        done.add(chosen)

        # update time: earliest resource free time (simple time advance)
        t = min(min(aM.values()), min(aL.values()))

    return S_local, C_local, mu, lam


# ===============================
#  PLOTTING (Gantt) for heuristic schedule
# ===============================

def plot_gantt_by_machine_heur(S, C, mu, title="Machine-wise schedule (heuristic)"):
    plt.figure()
    y_ticks, y_labels = [], []

    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        mtype = "TIG" if machine_type[m] == 1 else "MAG"
        y_labels.append(f"Machine {m} ({mtype})")

        ops = [i for i in I if mu[i] == m]
        ops = sorted(ops, key=lambda i: S[i])

        for i in ops:
            plt.barh(y_pos, C[i] - S[i], left=S[i])
            plt.text(S[i], y_pos, f"{i}", va="center", fontsize=7)

    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


def plot_gantt_by_station_heur(S, C, lam, title="Station-wise schedule (heuristic)"):
    fig, (ax, ax_leg) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [4, 1]}
    )

    cmap = plt.cm.get_cmap("tab10")
    job_colors = {j: cmap((j - 1) % 10) for j in J}

    y_ticks, y_labels = [], []

    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)
        y_labels.append("Station 1 (big station)" if l == 1 else f"Station {l}")

        ops = [i for i in I if lam[i] == l]
        ops = sorted(ops, key=lambda i: S[i])

        for i in ops:
            j = JOB_OF[i]
            ax.barh(y_pos, C[i] - S[i], left=S[i], color=job_colors[j])
            ax.text(S[i] + (C[i] - S[i]) / 2.0, y_pos, f"{i}",
                    ha="center", va="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    # legend panel
    ax_leg.set_axis_off()
    ax_leg.set_title("Job Color Key", fontsize=12, pad=10)

    y0 = 0.9
    dy = 0.7 / max(len(J), 1)
    for idx, j in enumerate(J):
        y_pos = y0 - idx * dy
        rect = plt.Rectangle((0.05, y_pos - 0.03), 0.12, 0.06,
                             transform=ax_leg.transAxes,
                             facecolor=job_colors[j], edgecolor="black")
        ax_leg.add_patch(rect)
        ax_leg.text(0.22, y_pos, f"Job {j}", transform=ax_leg.transAxes,
                    va="center", ha="left", fontsize=10)

    fig.tight_layout()


# ===============================
#  RUN
# ===============================

if __name__ == "__main__":
    print("=== HEURISTIC RUN (GT + ATCS) ===")
    print("Scenario: BIG station bottleneck + tight due dates => tardiness expected")

    # Build heuristic schedule
    S, C, assign_machine, assign_station = heuristic_schedule(k1=2.0)

    # Job-level completion
    C_weld = {}
    C_final = {}
    Tard = {}
    for j in J:
        last = O_j[j][-1]
        C_weld[j] = C[last]
        C_final[j] = C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j]
        Tard[j] = C_final[j] - d_j[j]  # can be negative

    Tmax = max(Tard.values())
    Cmax = max(C_final.values())

    # Print results
    print("\n===== Objective-like KPIs (heuristic evaluation) =====")
    print(f"T_max = {Tmax:.2f}")
    print(f"C_max = {Cmax:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld={C_weld[j]:.2f}, C_final={C_final[j]:.2f}, "
              f"T_j={Tard[j]:.2f}, d_j={d_j[j]:.2f}, r_j={r_j[j]:.2f}")

    print("\n===== Operations =====")
    for i in I:
        m_sel = assign_machine[i]
        l_sel = assign_station[i]
        mtype_str = "TIG" if machine_type[m_sel] == 1 else "MAG"
        print(f"Op {i}: S={S[i]:.2f}, C={C[i]:.2f}, machine={m_sel} ({mtype_str}), station={l_sel}")

    # Feasibility check
    check_heuristic_solution()

    # Gantt plots
    plot_gantt_by_machine_heur(S, C, assign_machine, title="Machine-wise welding schedule ")
    plot_gantt_by_station_heur(S, C, assign_station, title="Station-wise welding schedule ")
    plt.show()