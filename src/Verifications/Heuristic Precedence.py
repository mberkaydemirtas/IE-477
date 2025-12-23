#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 17:58:50 2025

@author: ezgieker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 17:43:15 2025
@author: ezgieker

Heuristic GT+ATCS scheduling with:
- Release/Due scatter
- Precedence matrix heatmap (like your screenshot)
- Machine-wise Gantt
- Station-wise Gantt
"""

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===============================
#  FIGURE SETTINGS (Gurobi-like)
# ===============================
FIGSIZE_GANTT = (10, 6)
FIGSIZE_DUE   = (6.4, 4.8)
FIGSIZE_PREC  = (6.4, 4.8)
FIG_DPI = 100

# ===============================
#  DATA (HARD PRECEDENCE TEST – 2 jobs, 6 ops)
# ===============================

J = [1, 2]
I = [1, 2, 3, 4, 5, 6]

O_j = {
    1: [1, 2, 3, 4],
    2: [5, 6],
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
L_i = {i: L[:] for i in I}

L_big   = [1, 2, 3]
L_small = [4, 5, 6, 7, 8, 9]

# ===============================
#  PRECEDENCE
# ===============================

Pred_i = {i: [] for i in I}

# Job 1: 1 -> 2, 1 -> 3, (2,3) -> 4
Pred_i[2] = [1]
Pred_i[3] = [1]
Pred_i[4] = [2, 3]

# Job 2: 5 -> 6
Pred_i[6] = [5]

# Cross-job: 4 -> 5
Pred_i[5] = [4]

# Optional: make each job's last operation depend on all ops in that job
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for op in ops:
        if op != last:
            preds.add(op)
    Pred_i[last] = list(preds)

# ===============================
#  PROCESSING TIMES
# ===============================
base_p = {1: 2.0, 2: 4.0, 3: 4.0, 4: 3.0, 5: 5.0, 6: 2.0}
p_im = {}
for i in I:
    for m in M_i[i]:
        p_im[(i, m)] = base_p[i] + 0.2 * (m - 1)

# Release / Due
r_j = {1: 0.0, 2: 0.0}
d_j = {1: 11.0, 2: 16.0}

# Grinding / Painting
g_j = {1: 1, 2: 0}
p_j = {1: 0, 2: 0}
t_grind_j = {1: 2.0, 2: 0.0}
t_paint_j = {1: 0.0, 2: 0.0}

# Big-station requirement: only job-end ops
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
#  PRECEDENCE MATRIX PLOT (NEW)
# ===============================

def plot_precedence_matrix(I_, Pred_i_, title="Precedence matrix for all operations"):
    """
    Matrix A where A[h,i] = 1 if h is a predecessor of i.
    Axes match your screenshot:
      x-axis: Operation i (successor)
      y-axis: Predecessor of operation i
    """
    I_sorted = list(I_)
    idx = {op: k for k, op in enumerate(I_sorted)}
    n = len(I_sorted)

    A = [[0 for _ in range(n)] for __ in range(n)]
    for succ in I_sorted:
        for pred in Pred_i_.get(succ, []):
            if pred in idx:
                A[idx[pred]][idx[succ]] = 1

    plt.figure(figsize=FIGSIZE_PREC, dpi=FIG_DPI)
    im = plt.imshow(A, aspect="auto")
    plt.colorbar(im, label="Precedence (1 → )")
    plt.title(title)
    plt.xlabel("Operation i")
    plt.ylabel("Predecessor of operation i")
    plt.xticks(range(n), I_sorted)
    plt.yticks(range(n), I_sorted)
    plt.tight_layout()

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
ALPHA_LOAD = 0.15
BETA_ST_LOAD = 0.05

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
            bestScore = float("inf")
            bestm = None
            for m in M_i[i]:
                st_m = max(t, R[i], avail_machine[m])
                cp_m = st_m + p_im[(i, m)]
                score = cp_m + ALPHA_LOAD * avail_machine[m]
                if score < bestScore:
                    bestScore = score
                    bestm = m

            best_machine[i] = bestm
            st_true = max(t, R[i], avail_machine[bestm])
            ecomp[i] = st_true + p_im[(i, bestm)]

        i_star = min(ready, key=lambda i: ecomp[i])
        m_star = best_machine[i_star]

        conflict = [i for i in ready if best_machine[i] == m_star]
        chosen = max(conflict, key=lambda i: atcs(i, t))

        m = best_machine[chosen]

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

    # NEW: Precedence matrix heatmap
    plot_precedence_matrix(I, Pred_i, title="Precedence matrix for all operations")

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
