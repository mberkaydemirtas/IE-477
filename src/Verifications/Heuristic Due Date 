#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 17:37:29 2025

@author: ezgieker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic GT+ATCS scheduling with Gurobi-like plots (balanced version):
- Due/Release scatter
- Machine-wise Gantt
- Station-wise Gantt

Key change vs your original:
✅ Adds a small LOAD-BALANCING penalty when picking the "best machine"
   so everything doesn't pile onto the fastest machine IDs.

You can tune:
- ALPHA_LOAD (machine load penalty)
- BETA_ST_LOAD (station load penalty)  (optional, small)
"""

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===============================
#  FIGURE SETTINGS (Gurobi-like)
# ===============================
FIGSIZE_GANTT = (10, 6)
FIGSIZE_DUE   = (6.4, 4.8)
FIG_DPI = 100

# ===============================
#  DATA – YOUR ORIGINAL (Urgent-job scenario)
# ===============================

J = [1, 2, 3, 4, 5, 6, 7, 8]
I = list(range(1, 31))

O_j = {
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    2: [11, 12, 13, 14],
    3: [15, 16, 17, 18],
    4: [19, 20],
    5: [21, 22, 23],
    6: [24, 25],
    7: [26, 27, 28],
    8: [29, 30],
}

# Machines
M = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
K = [1, 2]  # 1=TIG, 2=MAG

machine_type = {
    1: 1, 2: 1, 3: 2, 4: 2,
    5: 1, 6: 1, 7: 1, 8: 2,
    9: 1, 10: 1, 11: 1, 12: 1,
}

# Feasible machine types per op
K_i = {}
for i in I:
    r = i % 3
    if r == 1:
        K_i[i] = [1]
    elif r == 2:
        K_i[i] = [2]
    else:
        K_i[i] = [1, 2]

# Feasible machines per op
M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

# Stations
L = [1, 2, 3, 4, 5, 6, 7, 8, 9]
L_i = {i: [1, 2, 3, 4, 5, 6, 7, 8, 9] for i in I}

L_big   = [1, 2, 3]
L_small = [4, 5, 6, 7, 8, 9]

# Precedence
Pred_i = {i: [] for i in I}
for j in J:
    ops = O_j[j]
    for k in range(1, len(ops)):
        Pred_i[ops[k]].append(ops[k - 1])

# extra: last op after all ops in that job
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for i_op in ops:
        if i_op != last:
            preds.add(i_op)
    Pred_i[last] = list(preds)

# Processing times
p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)          # 3..7
        machine_add = (m - 1) * 0.5  # ⚠️ makes low-ID machines systematically faster
        p_im[(i, m)] = float(base + machine_add)

# Release times
r_j = {j: 0.0 for j in J}

# Due dates
d_j = {
    1: 25.0,
    2: 80.0,
    3: 90.0,
    4: 85.0,
    5: 100.0,
    6: 95.0,
    7: 110.0,
    8: 120.0,
}

# Grinding / Painting
g_j = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0}
p_j = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

# Big-station requirement: only end ops
Iend = [O_j[j][-1] for j in J]
beta_i = {i: (1 if i in Iend else 0) for i in I}

# Colors
_cmap = plt.cm.get_cmap("tab10")
JOB_COLORS = {j: _cmap((j - 1) % 10) for j in J}

# ===============================
#  HELPERS
# ===============================

def get_job_of_map(J_, O_j_):
    job_of = {}
    for jj in J_:
        for op in O_j_[jj]:
            job_of[op] = jj
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

    miss = [(i, m) for i in I for m in M_i[i] if (i, m) not in p_im]
    print("OK: p_im complete." if not miss else f"[ERROR] missing p_im {miss[:10]} ...")

    for i in I:
        if beta_i[i] == 1:
            feas = [l for l in L_i[i] if l not in L_small]
            if len(feas) == 0:
                raise ValueError(f"Op {i} requires BIG station but has no BIG station feasible!")
    print("OK: big-station feasibility consistent.")

# ===============================
#  DUE/RELEASE PLOT
# ===============================

def plot_release_due_dates_like_gurobi(J_, r_j_, d_j_):
    job_ids = list(J_)
    releases = [r_j_[j] for j in job_ids]
    dues     = [d_j_[j] for j in job_ids]

    plt.figure(figsize=FIGSIZE_DUE, dpi=FIG_DPI)
    plt.scatter(releases, job_ids, marker="o", label="release")
    plt.scatter(dues,     job_ids, marker="x", label="due")
    plt.xlabel("Time")
    plt.ylabel("Job")
    plt.title("Release and due dates per job")
    plt.yticks(job_ids)
    plt.legend()
    plt.tight_layout()

# ===============================
#  GANTT PLOTS
# ===============================

def plot_gantt_by_machine_like_gurobi(I_, M_, S_, C_, assign_machine_,
                                      title="Machine-wise welding schedule"):
    fig, ax = plt.subplots(figsize=FIGSIZE_GANTT, dpi=FIG_DPI)
    y_ticks, y_labels = [], []

    op_to_job = {}
    for j in J:
        for op in O_j[j]:
            op_to_job[op] = j

    for idx_m, m in enumerate(M_):
        y_pos = idx_m
        y_ticks.append(y_pos)

        mtype = "TIG" if machine_type[m] == 1 else "MAG"
        y_labels.append(f"Machine {m} ({mtype})")

        ops = [i for i in I_ if assign_machine_.get(i, None) == m]
        ops.sort(key=lambda i: S_[i])

        for i in ops:
            start = S_[i]
            finish = C_[i]
            width = finish - start
            color = JOB_COLORS.get(op_to_job.get(i, None), "gray")
            ax.barh(y_pos, width, left=start, color=color)
            ax.text(start + width / 2, y_pos, f"{i}",
                    va="center", ha="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [Patch(facecolor=JOB_COLORS[j], label=f"Job {j}") for j in J]
    ax.legend(handles=handles, title="Jobs",
              loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])

def plot_gantt_by_station_like_gurobi(I_, L_, S_, C_, assign_station_,
                                      title="Station-wise welding schedule"):
    fig, ax = plt.subplots(figsize=FIGSIZE_GANTT, dpi=FIG_DPI)
    y_ticks, y_labels = [], []

    op_to_job = {}
    for j in J:
        for op in O_j[j]:
            op_to_job[op] = j

    for idx_l, l in enumerate(L_):
        y_pos = idx_l
        y_ticks.append(y_pos)
        y_labels.append(f"Station {l}")

        ops = [i for i in I_ if assign_station_.get(i, None) == l]
        ops.sort(key=lambda i: S_[i])

        for i in ops:
            start = S_[i]
            finish = C_[i]
            width = finish - start
            color = JOB_COLORS.get(op_to_job.get(i, None), "gray")
            ax.barh(y_pos, width, left=start, color=color)
            ax.text(start + width / 2, y_pos, f"{i}",
                    va="center", ha="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [Patch(facecolor=JOB_COLORS[j], label=f"Job {j}") for j in J]
    ax.legend(handles=handles, title="Jobs",
              loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])

# ===============================
#  HEURISTIC: GT + ATCS (BALANCED)
# ===============================

# 👉 Tune these to reduce piling:
ALPHA_LOAD = 0.15     # machine-load penalty strength (0.05–0.30 good range)
BETA_ST_LOAD = 0.05   # station-load penalty (optional, keep small)

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

        # ===============================
        #  CHANGE 1: "best machine" now includes a load penalty
        # ===============================
        best_machine = {}
        ecomp = {}
        for i in ready:
            bestScore = float("inf")
            bestm = None
            for m in M_i[i]:
                st_m = max(t, R[i], avail_machine[m])
                cp_m = st_m + p_im[(i, m)]

                # ✅ load-aware score (keeps earliest completion, discourages piling)
                score = cp_m + ALPHA_LOAD * avail_machine[m]

                if score < bestScore:
                    bestScore = score
                    bestm = m

            best_machine[i] = bestm

            # keep ecomp for GT pivot as TRUE completion time (not the penalized score)
            st_true = max(t, R[i], avail_machine[bestm])
            ecomp[i] = st_true + p_im[(i, bestm)]

        i_star = min(ready, key=lambda i: ecomp[i])
        m_star = best_machine[i_star]

        conflict = [i for i in ready if best_machine[i] == m_star]
        chosen = max(conflict, key=lambda i: atcs(i, t))

        m = best_machine[chosen]

        # ===============================
        #  CHANGE 2 (optional): station pick also load-aware
        # ===============================
        stations = feasible_stations(chosen)
        l = min(stations, key=lambda ll: (avail_station[ll] + BETA_ST_LOAD * avail_station[ll]))

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
                raise AssertionError(f"Precedence violated: {h}->{i}")

    for i in I:
        j = job_of[i]
        if S[i] + tol < r_j[j]:
            raise AssertionError(f"Release violated: op {i}")

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

    plot_release_due_dates_like_gurobi(J, r_j, d_j)

    S, C, am, al, Cw, Cf, T, T_max, C_max = heuristic_GT_ATCS(k1=2.0)

    print("\n===== Objective (Heuristic GT+ATCS, balanced) =====")
    print(f"T_max = {T_max:.2f}")
    print(f"C_max = {C_max:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld={Cw[j]:.2f}, C_final={Cf[j]:.2f}, T_j={T[j]:.2f}, d_j={d_j[j]:.2f}")

    check_solution(S, C, am, al)

    plot_gantt_by_machine_like_gurobi(I, M, S, C, am, title="Machine-wise welding schedule")
    plot_gantt_by_station_like_gurobi(I, L, S, C, al, title="Station-wise welding schedule")

    plt.show()
