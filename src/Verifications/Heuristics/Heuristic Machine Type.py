# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 17:02:48 2025

@author: Dell
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welding Shop Scheduling – GIFFLER–THOMPSON + ATCS Heuristic
SCENARIO (UPDATED):
- 4 jobs, 30 operations
- 12 machines, 9 stations (3 big, 6 small)
- Ops 1–25 are FORCED to a small bottleneck machine set (like before)
- Ops 26–30 have wider flexibility
- Precedence: each job is a chain; last op depends on all previous ops of that job
- Big-station rule: only last operation of each job must use big stations (1–3)

VISUALS (UPDATED LIKE CODE 1):
- jobs have fixed colors
- Gantt by machine and station: colored by job + legend
- Job→Operation membership plot
"""

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ===============================
#  DATA – 4 jobs, 30 ops
# ===============================

J = [1, 2, 3, 4]
I = list(range(1, 31))

O_j = {
    1: list(range(1, 16)),     # ops 1–15
    2: list(range(16, 26)),    # ops 16–25
    3: [26, 27, 28],
    4: [29, 30],
}

# ===============================
#  MACHINES & STATIONS
# ===============================

M = list(range(1, 13))   # 12 machines: 1..12
L = list(range(1, 10))   # 9 stations: 1..9

L_big   = [1, 2, 3]
L_small = [4, 5, 6, 7, 8, 9]

# Stations: all allowed for all operations (big-station rule enforced later)
L_i = {i: L[:] for i in I}

# ===============================
#  MACHINE FEASIBILITY (M_i)
#  "Most ops forced to specific machines"
# ===============================

BOTTLENECK_MACHINES = [1, 2]  # bottleneck set (extreme)
M_i = {}

for i in I:
    if i <= 25:
        M_i[i] = BOTTLENECK_MACHINES[:]   # forced to 1–2
    else:
        mod = i % 4
        if mod == 1:
            M_i[i] = [1, 2, 3, 4, 5]
        elif mod == 2:
            M_i[i] = [2, 5, 6, 7, 8]
        elif mod == 3:
            M_i[i] = [1, 8, 9, 10]
        else:
            M_i[i] = [2, 9, 11, 12]

# ===============================
#  PRECEDENCE — chain per job + last op depends on all previous
# ===============================

Pred_i = {i: [] for i in I}

for j in J:
    ops = O_j[j]
    for k in range(1, len(ops)):
        Pred_i[ops[k]].append(ops[k - 1])

    last = ops[-1]
    preds = set(Pred_i[last])
    for op in ops[:-1]:
        preds.add(op)
    Pred_i[last] = list(preds)

# ===============================
#  PROCESSING TIMES p_im
# ===============================

p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)
        machine_offset = (m - 1) * 0.15
        p_im[(i, m)] = float(base + machine_offset)

# ===============================
#  RELEASE TIMES — all zero
# ===============================

r_j = {j: 0.0 for j in J}

# ===============================
#  DUE DATES — moderate
# ===============================

d_j = {1: 100.0, 2: 110.0, 3: 120.0, 4: 130.0}

# ===============================
#  Grinding/Painting flags and times
# ===============================

g_j = {1: 1, 2: 0, 3: 1, 4: 0}
p_flag_j = {1: 1, 2: 1, 3: 0, 4: 0}

t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    t_paint_j[j] = 3.0 if p_flag_j[j] == 1 else 0.0

# ===============================
#  BIG-STATION NEEDS (last ops only)
# ===============================

Iend = [O_j[j][-1] for j in J]
beta_i = {i: (1 if i in Iend else 0) for i in I}


# ==========================================================
#  HELPERS
# ==========================================================

def get_job_of_map(J, O_j):
    job_of = {}
    for j in J:
        for i in O_j[j]:
            job_of[i] = j
    return job_of


def get_job_colors(J):
    """
    Same logic as Code 1: fixed colors from tab10
    """
    cmap = plt.cm.get_cmap("tab10")
    return {j: cmap((j - 1) % 10) for j in J}


def verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j):
    print("\n=== DATA VERIFICATION ===")

    all_ops = set()
    for j in J:
        for op in O_j[j]:
            all_ops.add(op)
            if op not in I:
                print(f"[WARNING] Job {j} op {op} not in I")

    missing = set(I) - all_ops
    if missing:
        print(f"[WARNING] Ops in I but not in any job: {missing}")
    else:
        print("OK: All operations are covered by some job.")

    bad_m = [i for i in I if len(M_i[i]) == 0]
    bad_l = [i for i in I if len(L_i[i]) == 0]
    if bad_m:
        print(f"[ERROR] Ops with no feasible machines: {bad_m}")
    else:
        print("OK: Every op has feasible machines.")
    if bad_l:
        print(f"[ERROR] Ops with no feasible stations: {bad_l}")
    else:
        print("OK: Every op has feasible stations.")

    bad_pred = [(i, h) for i in I for h in Pred_i[i] if h not in I]
    if bad_pred:
        print(f"[ERROR] Invalid Pred entries: {bad_pred}")
    else:
        print("OK: All Pred_i references are valid.")

    missing_p = [(i, m) for i in I for m in M_i[i] if (i, m) not in p_im]
    if missing_p:
        print(f"[ERROR] Missing p_im for: {missing_p[:10]} ...")
    else:
        print("OK: All p_im entries exist.")

    if set(r_j.keys()) != set(J):
        print("[ERROR] r_j missing jobs!")
    else:
        print("OK: r_j defined for all jobs.")
    if set(d_j.keys()) != set(J):
        print("[ERROR] d_j missing jobs!")
    else:
        print("OK: d_j defined for all jobs.")


def visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_flag_j, t_grind_j, t_paint_j):
    print("\n=== JOB – OP SUMMARY ===")
    for j in J:
        ops = O_j[j]
        print(f"Job {j}: num_ops={len(ops)}, r_j={r_j[j]}, d_j={d_j[j]}, "
              f"grind={g_j[j]}, paint={p_flag_j[j]}, "
              f"t_grind={t_grind_j[j]}, t_paint={t_paint_j[j]}")

    plt.figure()
    plt.bar(J, [len(O_j[j]) for j in J])
    plt.xlabel("Job")
    plt.ylabel("Number of operations")
    plt.title("Number of operations per job")
    plt.tight_layout()

    plt.figure()
    plt.scatter(J, [r_j[j] for j in J], marker="o", label="release")
    plt.scatter(J, [d_j[j] for j in J], marker="x", label="due")
    plt.xlabel("Job")
    plt.ylabel("Time")
    plt.title("Release and due dates per job")
    plt.legend()
    plt.tight_layout()


def visualize_job_operation_membership(J, O_j):
    """
    Like Code 1: shows which operations belong to which job
    """
    job_colors = get_job_colors(J)

    plt.figure(figsize=(10, 4))
    for j in J:
        ops = O_j[j]
        plt.scatter(ops, [j] * len(ops), color=job_colors[j], label=f"Job {j}")
        for i in ops:
            plt.text(i, j + 0.05, str(i), ha="center", va="bottom", fontsize=7)

    plt.yticks(J, [f"Job {j}" for j in J])
    plt.xlabel("Global operation index (i)")
    plt.ylabel("Job")
    plt.title("Job → Operation membership")
    plt.grid(True, axis="x", linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()


def visualize_precedence_matrix(I, Pred_i):
    print("\n=== GLOBAL PRECEDENCE LIST (h -> i) ===")
    for i in I:
        for h in Pred_i[i]:
            print(f"{h} -> {i}")

    n = len(I)
    idx_of = {op: k for k, op in enumerate(I)}
    mat = [[0] * n for _ in range(n)]
    for i in I:
        ci = idx_of[i]
        for h in Pred_i[i]:
            rh = idx_of[h]
            mat[rh][ci] = 1

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Precedence (1 = h -> i)")
    plt.xticks(range(n), I, rotation=90)
    plt.yticks(range(n), I)
    plt.xlabel("Successor operation i")
    plt.ylabel("Predecessor operation h")
    plt.title("Precedence matrix for all operations")
    plt.tight_layout()


# ===============================
#  COLORED GANTT (LIKE CODE 1)
# ===============================

def plot_gantt_by_machine_colored(I, M, assign_machine, S, C,
                                  J, O_j,
                                  title="Machine-wise welding schedule (heuristic)"):
    """
    Single-machine Gantt (not segmented) – for small instances.
    Jobs colored + legend panel on right (like Code 1).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    job_of = get_job_of_map(J, O_j)
    job_colors = get_job_colors(J)

    y_ticks, y_labels = [], []
    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        y_labels.append(f"Machine {m}")

        for i in I:
            if assign_machine.get(i) != m:
                continue
            start = S[i]
            finish = C[i]
            width = finish - start
            j = job_of[i]
            ax.barh(y_pos, width, left=start, color=job_colors[j])
            ax.text(start + width / 2, y_pos, f"{i}", va="center", ha="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [Patch(facecolor=job_colors[j], label=f"Job {j}") for j in J]
    ax.legend(handles=handles, title="Jobs", loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.82, 1])


def plot_gantt_by_station_colored(I, L, assign_station, S, C,
                                 J, O_j, L_big,
                                 title="Station-wise welding schedule (heuristic)"):
    """
    Station Gantt with big/small labels, job colors + legend like Code 1.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    job_of = get_job_of_map(J, O_j)
    job_colors = get_job_colors(J)

    y_ticks, y_labels = [], []
    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)
        if l in L_big:
            y_labels.append(f"Station {l} (big)")
        else:
            y_labels.append(f"Station {l} (small)")

        for i in I:
            if assign_station.get(i) != l:
                continue
            start = S[i]
            finish = C[i]
            width = finish - start
            j = job_of[i]
            ax.barh(y_pos, width, left=start, color=job_colors[j])
            ax.text(start + width / 2, y_pos, f"{i}", va="center", ha="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [Patch(facecolor=job_colors[j], label=f"Job {j}") for j in J]
    ax.legend(handles=handles, title="Jobs", loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.82, 1])


# ==========================================================
#  HEURISTIC: GIFFLER–THOMPSON + ATCS
# ==========================================================

def heuristic_schedule(J, I, O_j, M, L,
                       M_i, L_i, Pred_i,
                       r_j, d_j,
                       g_j, p_flag_j,
                       t_grind_j, t_paint_j,
                       p_im, beta_i,
                       k1=2.0):

    job_of = get_job_of_map(J, O_j)

    all_p = [p_im[(i, m)] for i in I for m in M_i[i]]
    p_bar = sum(all_p) / len(all_p) if all_p else 1.0

    avail_machine = {m: 0.0 for m in M}
    avail_station = {l: 0.0 for l in L}

    S, C = {}, {}
    assign_machine, assign_station = {}, {}

    t = 0.0
    scheduled = set()

    def atcs_index(i, t_now):
        j = job_of[i]
        p_i = min(p_im[(i, m)] for m in M_i[i])
        slack = max(d_j[j] - p_i - t_now, 0.0)
        return (1.0 / p_i) * math.exp(-slack / (k1 * p_bar))

    while len(scheduled) < len(I):

        candidates = []
        R_i = {}

        for i in I:
            if i in scheduled:
                continue
            if any(h not in scheduled for h in Pred_i[i]):
                continue

            j = job_of[i]
            rt = r_j[j]
            if Pred_i[i]:
                rt = max(rt, max(C[h] for h in Pred_i[i]))
            R_i[i] = rt
            candidates.append(i)

        if not candidates:
            raise RuntimeError("No candidates found – precedence cycle or data issue?")

        ready = [i for i in candidates if R_i[i] <= t + 1e-9]
        if not ready:
            t = min(R_i[i] for i in candidates)
            ready = [i for i in candidates if R_i[i] <= t + 1e-9]

        earliest_completion = {}
        best_pair = {}

        for i in ready:
            bestC = float("inf")
            bestML = None

            machines_sorted = sorted(M_i[i], key=lambda mm: avail_machine[mm])
            stations_sorted = sorted(L_i[i], key=lambda ll: avail_station[ll])

            for m in machines_sorted:
                for l in stations_sorted:
                    if beta_i[i] == 1 and l in L_small:
                        continue
                    start = max(t, R_i[i], avail_machine[m], avail_station[l])
                    comp = start + p_im[(i, m)]
                    if comp < bestC:
                        bestC = comp
                        bestML = (m, l)

            if bestML is None:
                raise RuntimeError(f"Operation {i} has no feasible (machine, station) pair!")

            earliest_completion[i] = bestC
            best_pair[i] = bestML

        i_star = min(ready, key=lambda ii: earliest_completion[ii])
        m_star, l_star = best_pair[i_star]

        conflict_set = [i_star]
        for i in ready:
            if i == i_star:
                continue
            mi, li = best_pair[i]
            if (mi == m_star) or (li == l_star):
                conflict_set.append(i)

        chosen = max(conflict_set, key=lambda ii: atcs_index(ii, t))

        m_chosen, l_chosen = best_pair[chosen]
        start = max(t, R_i[chosen], avail_machine[m_chosen], avail_station[l_chosen])
        comp = start + p_im[(chosen, m_chosen)]

        S[chosen] = start
        C[chosen] = comp
        assign_machine[chosen] = m_chosen
        assign_station[chosen] = l_chosen

        avail_machine[m_chosen] = comp
        avail_station[l_chosen] = comp

        scheduled.add(chosen)
        t = min(min(avail_machine.values()), min(avail_station.values()))

    # job metrics
    C_weld, C_final, T = {}, {}, {}
    for j in J:
        last_op = O_j[j][-1]
        C_weld[j] = C[last_op]
        C_final[j] = C_weld[j] + g_j[j] * t_grind_j[j] + p_flag_j[j] * t_paint_j[j]
        T[j] = C_final[j] - d_j[j]

    C_max = max(C_final.values())
    T_max = max(T.values())

    return S, C, assign_machine, assign_station, C_weld, C_final, T, T_max, C_max


def check_heuristic_solution(J, I, O_j, M, L,
                             Pred_i, r_j,
                             S, C, assign_machine, assign_station,
                             beta_i,
                             tol=1e-6):

    print("\n=== CHECKING HEURISTIC SOLUTION ===")
    job_of = get_job_of_map(J, O_j)

    for i in I:
        for h in Pred_i[i]:
            if S[i] + tol < C[h]:
                raise AssertionError(f"Precedence violated: op {i} starts {S[i]:.3f} < C[{h}]={C[h]:.3f}")

    for i in I:
        j = job_of[i]
        if S[i] + tol < r_j[j]:
            raise AssertionError(f"Release violated: op {i} starts {S[i]:.3f} < r_j[{j}]={r_j[j]:.3f}")

    for m in M:
        ops_m = [i for i in I if assign_machine.get(i) == m]
        for a in range(len(ops_m)):
            for b in range(a + 1, len(ops_m)):
                i, h = ops_m[a], ops_m[b]
                if not (C[i] <= S[h] + tol or C[h] <= S[i] + tol):
                    raise AssertionError(f"Machine {m} overlap: ops {i} and {h}")

    for l in L:
        ops_l = [i for i in I if assign_station.get(i) == l]
        for a in range(len(ops_l)):
            for b in range(a + 1, len(ops_l)):
                i, h = ops_l[a], ops_l[b]
                if not (C[i] <= S[h] + tol or C[h] <= S[i] + tol):
                    raise AssertionError(f"Station {l} overlap: ops {i} and {h}")

    for i in I:
        if beta_i[i] == 1 and assign_station[i] in L_small:
            raise AssertionError(f"Big-station violated: op {i} assigned to small station {assign_station[i]}")

    print("All checks passed.\n")


# ===============================
#  MAIN
# ===============================
if __name__ == "__main__":

    verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j)
    visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_flag_j, t_grind_j, t_paint_j)
    visualize_job_operation_membership(J, O_j)
    visualize_precedence_matrix(I, Pred_i)

    (S, C,
     assign_machine, assign_station,
     C_weld, C_final, T,
     T_max, C_max) = heuristic_schedule(
        J, I, O_j,
        M, L,
        M_i, L_i, Pred_i,
        r_j, d_j,
        g_j, p_flag_j,
        t_grind_j, t_paint_j,
        p_im, beta_i,
        k1=2.0
    )

    check_heuristic_solution(
        J, I, O_j,
        M, L,
        Pred_i, r_j,
        S, C, assign_machine, assign_station,
        beta_i,
        tol=1e-6
    )

    print("===== Objective (Heuristic) =====")
    print(f"T_max = {T_max:.2f}")
    print(f"C_max = {C_max:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld={C_weld[j]:.2f}, C_final={C_final[j]:.2f}, "
              f"T_j={T[j]:.2f}, d_j={d_j[j]:.2f}")

    print("\n===== Operations =====")
    for i in I:
        print(f"Op {i}: S={S[i]:.2f}, C={C[i]:.2f}, machine={assign_machine[i]}, station={assign_station[i]}")

    # ✅ COLORED GANTT (jobs visible like Code 1)
    plot_gantt_by_machine_colored(
        I, M, assign_machine, S, C,
        J, O_j,
        title="Machine-wise welding schedule (heuristic) – job-colored"
    )
    plot_gantt_by_station_colored(
        I, L, assign_station, S, C,
        J, O_j, L_big,
        title="Station-wise welding schedule (heuristic) – job-colored"
    )

    plt.show()