# -*- coding: utf-8 -*-
"""
HeuristicPlots.py
Reusable verification + visualization for heuristic scenarios.

Consumes `data` dict and (for gantt) `HeuristicResult`.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# -------------------------
# Helpers
# -------------------------

def get_job_of_map(J, O_j):
    job_of = {}
    for j in J:
        for i in O_j[j]:
            job_of[i] = j
    return job_of


def get_job_colors(J):
    cmap = plt.cm.get_cmap("tab10")
    return {j: cmap((j - 1) % 10) for j in J}


# -------------------------
# Verification (same prints)
# -------------------------

def verify_data(data):
    J = data["J"]
    I = data["I"]
    O_j = data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    r_j = data["r_j"]
    d_j = data["d_j"]

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
        print(f"[ERROR] Invalid Pred entries: {bad_pred[:10]} ...")
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


# -------------------------
# Summary plots (same as old)
# -------------------------

def visualize_jobs_and_ops(data):
    J = data["J"]
    O_j = data["O_j"]
    r_j = data["r_j"]
    d_j = data["d_j"]
    g_j = data["g_j"]
    p_flag_j = data["p_flag_j"]
    t_grind_j = data["t_grind_j"]
    t_paint_j = data["t_paint_j"]

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


def visualize_job_operation_membership(data):
    J = data["J"]
    O_j = data["O_j"]
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


def visualize_precedence_matrix(data):
    I = data["I"]
    Pred_i = data["Pred_i"]

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


# -------------------------
# Gantt plots (job-colored)
# -------------------------

def plot_gantt_by_machine_colored(data, res, title="Machine-wise welding schedule (heuristic) – job-colored"):
    I = data["I"]
    M = data["M"]
    J = data["J"]
    O_j = data["O_j"]

    fig, ax = plt.subplots(figsize=(12, 7))

    job_of = get_job_of_map(J, O_j)
    job_colors = get_job_colors(J)

    y_ticks, y_labels = [], []
    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        y_labels.append(f"Machine {m}")

        for i in I:
            if res.assign_machine.get(i) != m:
                continue
            start = res.S[i]
            finish = res.C[i]
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


def plot_gantt_by_station_colored(data, res, title="Station-wise welding schedule (heuristic) – job-colored"):
    I = data["I"]
    L = data["L"]
    J = data["J"]
    O_j = data["O_j"]
    L_big = data["L_big"]

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
            if res.assign_station.get(i) != l:
                continue
            start = res.S[i]
            finish = res.C[i]
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
