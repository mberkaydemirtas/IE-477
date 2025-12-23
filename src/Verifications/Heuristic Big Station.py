#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:13:45 2025

@author: ezgieker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===============================
#  FIGURE SIZES (match your screenshots)
# ===============================
FIGSIZE_MACHINE = (7.0, 3.0)   # small machine chart (like your 1st screenshot)
FIGSIZE_STATION = (14.0, 6.0)  # big station chart (like your 2nd screenshot)
FIG_DPI = 120

# ===============================
#  DATA (LARGE EXAMPLE – 8 jobs, 30 ops)
# ===============================

J = [1, 2, 3, 4, 5, 6, 7, 8]
I = list(range(1, 31))  # 1..30

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

# ===============================
#  MACHINES (12 pcs, 2 TYPES) & STATIONS
# ===============================

M = list(range(1, 13))  # 1..12
K = [1, 2]

machine_type = {
    1: 1, 2: 1, 3: 1,       # TIG
    4: 2, 5: 2, 6: 2, 7: 2,
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2
}

K_i = {}
for i in I:
    K_i[i] = [1] if i % 2 == 1 else [2]

M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

L = [1, 2, 3, 4]
L_i = {i: [1, 2, 3, 4] for i in I}
L_big = [1]
L_small = [2, 3, 4]

# ===============================
#  PRECEDENCE
# ===============================

Pred_i = {i: [] for i in I}

Pred_i[2] = [1]
Pred_i[3] = [1]

Pred_i[5] = [4]
Pred_i[6] = [4]
Pred_i[7] = [5]

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

# Job 8: no internal precedence

# Each job's last op after all ops in that job
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for i_op in ops:
        if i_op != last:
            preds.add(i_op)
    Pred_i[last] = list(preds)

# ===============================
#  PROCESSING TIMES p_im
# ===============================

p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)          # 3..7
        machine_add = (m - 1) * 0.5
        p_im[(i, m)] = float(base + machine_add)

# ===============================
#  RELEASE TIMES r_j
# ===============================

r_j = {}
release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
for idx, j in enumerate(J):
    r_j[j] = release_times[idx]

# ===============================
#  DUE DATES d_j
# ===============================

d_i = {}
due_base = 30.0
Iend = [O_j[j][-1] for j in J]
for idx, i_last in enumerate(Iend):
    d_i[i_last] = due_base + 3.0 * idx  # 30,33,...,51

d_j = {j: d_i[O_j[j][-1]] for j in J}

# If you want: make job 8 very tight (bariz tardy)
# d_j[8] = 25.0

# ===============================
#  GRINDING & PAINTING
# ===============================

g_j = {j: (1 if (j-1) % 2 == 0 else 0) for j in J}  # j=1,3,5,7 => 1
p_j = {j: (1 if (j in [1, 2, 3]) else 0) for j in J}

t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

# ===============================
#  BIG-STATION REQUIREMENT beta_i
# ===============================

beta_i = {}
first_ops = [O_j[j][0] for j in J]
for i in I:
    beta_i[i] = 0 if i in first_ops else 1

# ===============================
#  JOB COLORS (same palette as your example)
# ===============================

JOB_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
    6: "tab:brown",
    7: "tab:pink",
    8: "tab:gray",
}

# ===============================
#  HELPERS
# ===============================

def get_job_of_map(J, O_j):
    job_of = {}
    for j in J:
        for i in O_j[j]:
            job_of[i] = j
    return job_of

def verify_data():
    print("\n=== DATA VERIFICATION ===")
    all_ops = set().union(*[set(O_j[j]) for j in J])
    if set(I) != all_ops:
        print("[WARNING] I and union(O_j) mismatch!")
        print("Missing in O_j:", sorted(set(I) - all_ops))
        print("Extra in O_j  :", sorted(all_ops - set(I)))
    else:
        print("OK: I matches union(O_j).")

    badM = [i for i in I if len(M_i[i]) == 0]
    badL = [i for i in I if len(L_i[i]) == 0]
    print("OK: machines feasible for all ops." if not badM else f"[ERROR] no machine for {badM}")
    print("OK: stations feasible for all ops." if not badL else f"[ERROR] no station for {badL}")

    bad_pred = [(i, h) for i in I for h in Pred_i[i] if h not in I]
    print("OK: Pred_i references are valid." if not bad_pred else f"[ERROR] bad preds {bad_pred}")

    miss = [(i, m) for i in I for m in M_i[i] if (i, m) not in p_im]
    print("OK: p_im complete." if not miss else f"[ERROR] missing p_im {miss[:10]} ...")

    if set(r_j.keys()) == set(J):
        print("OK: r_j defined for all jobs.")
    else:
        print("[ERROR] r_j missing keys:", set(J) - set(r_j.keys()))

    if set(d_j.keys()) == set(J):
        print("OK: d_j defined for all jobs.")
    else:
        print("[ERROR] d_j missing keys:", set(J) - set(d_j.keys()))

    for i in I:
        if beta_i[i] == 1:
            feas = [l for l in L_i[i] if l not in L_small]
            if len(feas) == 0:
                raise ValueError(f"Op {i} requires big station but has no big station feasible!")
    print("OK: big-station feasibility consistent.")

# ===============================
#  COLORED GANTT PLOTS (with your desired sizes)
# ===============================

def _legend_job_key():
    return [Patch(facecolor=JOB_COLORS[j], edgecolor="black", label=f"Job {j}") for j in sorted(JOB_COLORS)]

def plot_gantt_by_machine_colored(I, M, S, C, assign_machine, job_of,
                                  title="Machine-wise welding schedule",
                                  figsize=FIGSIZE_MACHINE, dpi=FIG_DPI):
    plt.figure(figsize=figsize, dpi=dpi)
    y_ticks, y_labels = [], []

    for idx_m, m in enumerate(M):
        y_ticks.append(idx_m)
        mtype = "TIG" if machine_type[m] == 1 else "MAG"
        y_labels.append(f"Machine {m} ({mtype})")

        ops = [i for i in I if assign_machine.get(i, None) == m]
        ops.sort(key=lambda i: S[i])

        for i in ops:
            j = job_of[i]
            color = JOB_COLORS.get(j, "gray")
            plt.barh(idx_m, C[i] - S[i], left=S[i],
                     color=color, edgecolor="black", linewidth=0.4)
            plt.text(S[i], idx_m, str(i), va="center", fontsize=7, color="black")

    plt.yticks(y_ticks, y_labels, fontsize=8)
    plt.xlabel("Time", fontsize=9)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.legend(handles=_legend_job_key(), title="Job Color Key",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=8, title_fontsize=9)

def plot_gantt_by_station_colored(I, L, S, C, assign_station, job_of,
                                  L_big=None,
                                  title="Station-wise welding schedule",
                                  figsize=FIGSIZE_STATION, dpi=FIG_DPI):
    if L_big is None:
        L_big = []

    plt.figure(figsize=figsize, dpi=dpi)
    y_ticks, y_labels = [], []

    for idx_l, l in enumerate(L):
        y_ticks.append(idx_l)
        y_labels.append(f"Station {l} (big station)" if l in L_big else f"Station {l}")

        ops = [i for i in I if assign_station.get(i, None) == l]
        ops.sort(key=lambda i: S[i])

        for i in ops:
            j = job_of[i]
            color = JOB_COLORS.get(j, "gray")
            plt.barh(idx_l, C[i] - S[i], left=S[i],
                     color=color, edgecolor="black", linewidth=0.4)
            plt.text(S[i], idx_l, str(i), va="center", fontsize=9, color="black")

    plt.yticks(y_ticks, y_labels, fontsize=10)
    plt.xlabel("Time", fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.legend(handles=_legend_job_key(), title="Job Color Key",
               bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=10, title_fontsize=11)

# ===============================
#  HEURISTIC: GT + ATCS (LESS OPTIMIZATION INSIDE)
# ===============================

def heuristic_GT_ATCS(k1=2.0):
    job_of = get_job_of_map(J, O_j)

    p_list = [min(p_im[(i, m)] for m in M_i[i]) for i in I]
    p_bar = sum(p_list) / len(p_list) if p_list else 1.0

    avail_machine = {m: 0.0 for m in M}
    avail_station = {l: 0.0 for l in L}

    S, C = {}, {}
    assign_machine, assign_station = {}, {}

    scheduled = set()
    t = 0.0

    def release_of(i):
        j = job_of[i]
        rt = r_j[j]
        if Pred_i[i]:
            rt = max(rt, max(C[h] for h in Pred_i[i]))
        return rt

    def atcs(i, t_now):
        j = job_of[i]
        p_i = min(p_im[(i, m)] for m in M_i[i])
        slack = max(d_j[j] - p_i - t_now, 0.0)
        return (1.0 / p_i) * math.exp(-slack / (k1 * p_bar))

    def feasible_stations(i):
        if beta_i[i] == 1:
            return [l for l in L_i[i] if l not in L_small]
        return list(L_i[i])

    while len(scheduled) < len(I):

        eligible = [i for i in I if i not in scheduled and all(h in scheduled for h in Pred_i[i])]
        if not eligible:
            raise RuntimeError("No eligible ops (cycle?)")

        R = {i: release_of(i) for i in eligible}
        ready = [i for i in eligible if R[i] <= t + 1e-9]

        if not ready:
            t = min(R[i] for i in eligible)
            ready = [i for i in eligible if R[i] <= t + 1e-9]

        best_machine = {}
        ecomp = {}
        for i in ready:
            bestC = float("inf")
            bestm = None
            for m in M_i[i]:
                st_m = max(t, R[i], avail_machine[m])
                cp_m = st_m + p_im[(i, m)]
                if cp_m < bestC:
                    bestC = cp_m
                    bestm = m
            best_machine[i] = bestm
            ecomp[i] = bestC

        i_star = min(ready, key=lambda i: ecomp[i])
        m_star = best_machine[i_star]

        conflict = [i for i in ready if best_machine[i] == m_star]
        chosen = max(conflict, key=lambda i: atcs(i, t))

        m = best_machine[chosen]
        stations = feasible_stations(chosen)
        l = min(stations, key=lambda ll: avail_station[ll])

        start = max(t, R[chosen], avail_machine[m], avail_station[l])
        comp = start + p_im[(chosen, m)]

        S[chosen] = start
        C[chosen] = comp
        assign_machine[chosen] = m
        assign_station[chosen] = l

        avail_machine[m] = comp
        avail_station[l] = comp
        scheduled.add(chosen)

        next_times = []
        next_times += [tt for tt in avail_machine.values() if tt > t + 1e-9]
        next_times += [tt for tt in avail_station.values() if tt > t + 1e-9]

        remaining = [i for i in I if i not in scheduled and all(h in scheduled for h in Pred_i[i])]
        for i in remaining:
            rr = release_of(i)
            if rr > t + 1e-9:
                next_times.append(rr)

        if next_times:
            t = min(next_times)

    C_weld, C_final, T = {}, {}, {}
    for j in J:
        last = O_j[j][-1]
        C_weld[j] = C[last]
        C_final[j] = C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j]
        T[j] = C_final[j] - d_j[j]

    return S, C, assign_machine, assign_station, C_weld, C_final, T, max(T.values()), max(C_final.values())

def check_solution(S, C, assign_machine, assign_station, tol=1e-6):
    job_of = get_job_of_map(J, O_j)

    for i in I:
        for h in Pred_i[i]:
            if S[i] + tol < C[h]:
                raise AssertionError(f"Precedence violated: {h}->{i} (S[{i}]={S[i]:.2f} < C[{h}]={C[h]:.2f})")

    for i in I:
        j = job_of[i]
        if S[i] + tol < r_j[j]:
            raise AssertionError(f"Release violated: op {i} starts {S[i]:.2f} < r_j[{j}]={r_j[j]:.2f}")

    for i in I:
        if beta_i[i] == 1 and assign_station[i] in L_small:
            raise AssertionError(f"Big-station violated: op {i} on station {assign_station[i]}")

    for m in M:
        ops = [i for i in I if assign_machine[i] == m]
        ops.sort(key=lambda i: S[i])
        for a, b in zip(ops, ops[1:]):
            if C[a] > S[b] + tol:
                raise AssertionError(f"Machine overlap on {m}: ops {a},{b}")

    for l in L:
        ops = [i for i in I if assign_station[i] == l]
        ops.sort(key=lambda i: S[i])
        for a, b in zip(ops, ops[1:]):
            if C[a] > S[b] + tol:
                raise AssertionError(f"Station overlap on {l}: ops {a},{b}")

    print("All checks passed.")

# ===============================
#  MAIN
# ===============================

if __name__ == "__main__":
    verify_data()

    S, C, am, al, Cw, Cf, T, T_max, C_max = heuristic_GT_ATCS(k1=2.0)

    print("\n===== Objective (Heuristic GT+ATCS) =====")
    print(f"T_max = {T_max:.2f}")
    print(f"C_max = {C_max:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld={Cw[j]:.2f}, C_final={Cf[j]:.2f}, T_j={T[j]:.2f}, d_j={d_j[j]:.2f}")

    check_solution(S, C, am, al)

    job_of = get_job_of_map(J, O_j)

    plot_gantt_by_machine_colored(
        I, M, S, C, am, job_of,
        title="Machine-wise welding schedule",
        figsize=FIGSIZE_MACHINE, dpi=FIG_DPI
    )
    plot_gantt_by_station_colored(
        I, L, S, C, al, job_of,
        L_big=L_big,
        title="Station-wise welding schedule",
        figsize=FIGSIZE_STATION, dpi=FIG_DPI
    )

    plt.show()
