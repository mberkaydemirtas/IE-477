#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 12:55:41 2025

@author: ezgieker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welding Shop Scheduling – HARD PRECEDENCE, Tardiness > 0 example (HEURISTIC)

✅ Same DATA as your Gurobi model:
- 2 jobs, 6 operations
- 12 machines with types (TIG/MAG)
- 9 stations (1–3 big, 4–9 small)
- Tight due dates => tardiness > 0

✅ Replaces the whole Gurobi MIP with:
- GT (Giffler–Thompson) + ATCS dispatching heuristic
- Enforces: precedence + machine capacity + station capacity + big-station requirement
- Produces: S_h[i], C_h[i], assigned machine/station
- Prints: job-level completion + tardiness + T_max, C_max
- Plots: ops/job, release-due, precedence matrix, machine Gantt, station Gantt
"""

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===============================
#  DATA (same as your model)
# ===============================

J = [1, 2]
I = [1, 2, 3, 4, 5, 6]

O_j = {
    1: [1, 2, 3, 4],
    2: [5, 6],
}

M = list(range(1, 13))
K = [1, 2]  # 1=TIG, 2=MAG

machine_type = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,
}

K_i = {}
for i in I:
    r = i % 3
    if r == 1:
        K_i[i] = [1]
    elif r == 2:
        K_i[i] = [2]
    else:
        K_i[i] = [1, 2]

M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

L = list(range(1, 10))
L_i = {i: L[:] for i in I}

L_big = [1, 2, 3]
L_small = [4, 5, 6, 7, 8, 9]

Pred_i = {i: [] for i in I}
Pred_i[2] = [1]
Pred_i[3] = [1]
Pred_i[4] = [2, 3]
Pred_i[6] = [5]
Pred_i[5] = [4]

# keep your "last op depends on all ops in that job"
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for i_op in ops:
        if i_op != last:
            preds.add(i_op)
    Pred_i[last] = list(preds)

base_p = {1: 2.0, 2: 4.0, 3: 4.0, 4: 3.0, 5: 5.0, 6: 2.0}
p_im = {(i, m): base_p[i] + 0.2 * (m - 1) for i in I for m in M_i[i]}

r_j = {1: 0.0, 2: 0.0}
d_j = {1: 5.0, 2: 8.0}  # very tight

g_j = {1: 1, 2: 0}
p_j = {1: 0, 2: 0}
t_grind_j = {1: 2.0, 2: 0.0}
t_paint_j = {1: 0.0, 2: 0.0}

Iend = [O_j[j][-1] for j in J]  # [4,6]
beta_i = {i: (1 if i in Iend else 0) for i in I}

# Colors for jobs
_cmap = plt.cm.get_cmap("tab10")
JOB_COLORS = {j: _cmap((j - 1) % 10) for j in J}

# ===============================
#  VERIFICATION + VISUALS
# ===============================

def verify_data():
    print("\n=== DATA VERIFICATION ===")

    all_ops_from_jobs = set().union(*[set(O_j[j]) for j in J])
    if set(I) != all_ops_from_jobs:
        print("[WARNING] I and union(O_j) mismatch!")
        print("Missing in O_j:", sorted(set(I) - all_ops_from_jobs))
        print("Extra in O_j  :", sorted(all_ops_from_jobs - set(I)))
    else:
        print("OK: All operations in I are covered by some O_j.")

    bad_m = [i for i in I if len(M_i[i]) == 0]
    bad_l = [i for i in I if len(L_i[i]) == 0]
    print("OK: Every operation has at least one feasible machine." if not bad_m else f"[ERROR] {bad_m}")
    print("OK: Every operation has at least one feasible station." if not bad_l else f"[ERROR] {bad_l}")

    missing_p = [(i, m) for i in I for m in M_i[i] if (i, m) not in p_im]
    print("OK: p_im is complete." if not missing_p else f"[ERROR] missing p_im {missing_p[:10]}")

    for i in I:
        for h in Pred_i[i]:
            if h not in I:
                raise ValueError(f"[ERROR] Pred_i has invalid predecessor {h} for op {i}")

    if set(r_j.keys()) != set(J):
        print("[ERROR] r_j missing jobs")
    else:
        print("OK: r_j defined for all jobs.")
    if set(d_j.keys()) != set(J):
        print("[ERROR] d_j missing jobs")
    else:
        print("OK: d_j defined for all jobs.")

def visualize_jobs_and_ops():
    print("\n=== JOB – OP SUMMARY ===")
    for j in J:
        print(f"Job {j}: ops={O_j[j]}, r={r_j[j]}, d={d_j[j]}, grind={g_j[j]}, paint={p_j[j]}")

    plt.figure()
    plt.bar(J, [len(O_j[j]) for j in J])
    plt.xlabel("Job")
    plt.ylabel("Number of operations")
    plt.title("Number of operations per job")
    plt.tight_layout()

    job_ids = J
    releases = [r_j[j] for j in J]
    dues = [d_j[j] for j in J]

    plt.figure()
    plt.scatter(releases, job_ids, marker="o", label="release")
    plt.scatter(dues, job_ids, marker="x", label="due")
    plt.xlabel("Time")
    plt.ylabel("Job")
    plt.yticks(job_ids, [f"Job {j}" for j in J])
    plt.title("Release and due dates per job")
    plt.legend()
    plt.tight_layout()

def visualize_precedence_matrix():
    print("\n=== GLOBAL PRECEDENCE LIST (h -> i) ===")
    for i in I:
        for h in Pred_i[i]:
            print(f"{h} -> {i}")

    n = len(I)
    index_of = {op: idx for idx, op in enumerate(I)}
    mat = [[0] * n for _ in range(n)]
    for i in I:
        ci = index_of[i]
        for h in Pred_i[i]:
            rh = index_of[h]
            mat[rh][ci] = 1

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Precedence (1 = h -> i)")
    plt.xticks(range(n), I)
    plt.yticks(range(n), I)
    plt.xlabel("Operation i")
    plt.ylabel("Predecessor h")
    plt.title("Precedence matrix for all operations")
    plt.tight_layout()

def plot_gantt_by_machine(S, C, am, title="Machine-wise welding schedule (Heuristic)"):
    fig, ax = plt.subplots(figsize=(9, 4))
    y_ticks, y_labels = [], []

    op_to_job = {op: j for j in J for op in O_j[j]}

    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        mtype_str = "TIG" if machine_type[m] == 1 else "MAG"
        y_labels.append(f"Machine {m} ({mtype_str})")

        ops = [i for i in I if am.get(i) == m]
        ops.sort(key=lambda i: S[i])

        for i in ops:
            start = S[i]
            finish = C[i]
            width = finish - start
            color = JOB_COLORS[op_to_job[i]]
            ax.barh(y_pos, width, left=start, color=color)
            ax.text(start + width / 2, y_pos, f"{i}", va="center", ha="center", fontsize=8)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [Patch(facecolor=JOB_COLORS[j], label=f"Job {j}") for j in J]
    ax.legend(handles=handles, title="Jobs", loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])

def plot_gantt_by_station(S, C, al, title="Station-wise welding schedule (Heuristic)"):
    fig, ax = plt.subplots(figsize=(9, 4))
    y_ticks, y_labels = [], []

    op_to_job = {op: j for j in J for op in O_j[j]}

    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)
        y_labels.append(f"Station {l} ({'big' if l in L_big else 'small'})")

        ops = [i for i in I if al.get(i) == l]
        ops.sort(key=lambda i: S[i])

        for i in ops:
            start = S[i]
            finish = C[i]
            width = finish - start
            color = JOB_COLORS[op_to_job[i]]
            ax.barh(y_pos, width, left=start, color=color)
            ax.text(start + width / 2, y_pos, f"{i}", va="center", ha="center", fontsize=8)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [Patch(facecolor=JOB_COLORS[j], label=f"Job {j}") for j in J]
    ax.legend(handles=handles, title="Jobs", loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])

# ===============================
#  HEURISTIC: GT + ATCS
#  (Important fix: time advances to NEXT EVENT, not min(avail) which returns 0!)
# ===============================

def heuristic_gt_atcs(k1=2.0):
    eps = 1e-9
    op_to_job = {op: j for j in J for op in O_j[j]}

    # pbar = average over all feasible pairs
    allp = [p_im[(i, m)] for i in I for m in M_i[i]]
    pbar = sum(allp) / max(len(allp), 1)

    a_m = {m: 0.0 for m in M}  # machine availability
    a_l = {l: 0.0 for l in L}  # station availability

    S, C = {}, {}
    am, al = {}, {}
    done = set()
    t = 0.0

    def ready_time(i):
        j = op_to_job[i]
        pred_finish = max((C[h] for h in Pred_i[i]), default=0.0)
        return max(r_j[j], pred_finish)

    def feasible_stations(i):
        if beta_i[i] == 1:
            return [l for l in L_i[i] if l in L_big]
        return list(L_i[i])

    def atcs(i, t_now):
        j = op_to_job[i]
        p_i = min(p_im[(i, m)] for m in M_i[i])
        slack = max(d_j[j] - p_i - t_now, 0.0)
        return (1.0 / max(p_i, 1e-9)) * math.exp(-slack / max(k1 * pbar, 1e-9))

    while len(done) < len(I):
        # eligible by precedence
        eligible = [i for i in I if i not in done and all(h in done for h in Pred_i[i])]
        if not eligible:
            raise RuntimeError("No eligible operations. Check precedence for cycles.")

        R = {i: ready_time(i) for i in eligible}
        ready = [i for i in eligible if R[i] <= t + eps]
        if not ready:
            t = min(R[i] for i in eligible)
            ready = [i for i in eligible if R[i] <= t + eps]

        # for each ready op: best (m,l) minimizing completion
        best = {}
        ecomp = {}

        for i in ready:
            best_c = float("inf")
            best_tuple = None
            for m in M_i[i]:
                for l in feasible_stations(i):
                    s = max(t, R[i], a_m[m], a_l[l])
                    c = s + p_im[(i, m)]
                    if c < best_c:
                        best_c = c
                        best_tuple = (m, l, s, c)
            best[i] = best_tuple
            ecomp[i] = best_tuple[3]

        # GT pivot
        i_star = min(ready, key=lambda i: ecomp[i])
        m_star, l_star, _, _ = best[i_star]

        # conflict set: shares pivot machine or station
        conflict = []
        for i in ready:
            mi, li, _, _ = best[i]
            if (mi == m_star) or (li == l_star) or (i == i_star):
                conflict.append(i)

        chosen = max(conflict, key=lambda i: atcs(i, t))
        m_ch, l_ch, s_ch, c_ch = best[chosen]

        S[chosen] = s_ch
        C[chosen] = c_ch
        am[chosen] = m_ch
        al[chosen] = l_ch

        a_m[m_ch] = c_ch
        a_l[l_ch] = c_ch
        done.add(chosen)

        # advance time to next event strictly > t
        next_times = []
        next_times += [tt for tt in a_m.values() if tt > t + eps]
        next_times += [tt for tt in a_l.values() if tt > t + eps]

        # future releases among now-eligible-after-done ops
        still_eligible = [ii for ii in I if ii not in done and all(h in done for h in Pred_i[ii])]
        for ii in still_eligible:
            rr = ready_time(ii)
            if rr > t + eps:
                next_times.append(rr)

        if next_times:
            t = min(next_times)

    # job metrics
    C_weld = {}
    C_final = {}
    T = {}
    for j in J:
        last = O_j[j][-1]
        C_weld[j] = C[last]
        C_final[j] = C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j]
        T[j] = max(C_final[j] - d_j[j], 0.0)

    T_max = max(T.values())
    C_max = max(C_final.values())

    return S, C, am, al, C_weld, C_final, T, T_max, C_max

def check_solution_heuristic(S, C, am, al, tol=1e-6):
    # precedence
    for i in I:
        for h in Pred_i[i]:
            if S[i] + tol < C[h]:
                raise AssertionError(f"Precedence violated: {h}->{i}")

    # big-station
    for i in I:
        if beta_i[i] == 1 and al[i] in L_small:
            raise AssertionError(f"Big-station violated: op {i} on station {al[i]}")

    # machine overlap
    for m in M:
        ops = [i for i in I if am[i] == m]
        ops.sort(key=lambda i: S[i])
        for a, b in zip(ops, ops[1:]):
            if C[a] > S[b] + tol:
                raise AssertionError(f"Machine overlap on {m}: ops {a},{b}")

    # station overlap
    for l in L:
        ops = [i for i in I if al[i] == l]
        ops.sort(key=lambda i: S[i])
        for a, b in zip(ops, ops[1:]):
            if C[a] > S[b] + tol:
                raise AssertionError(f"Station overlap on {l}: ops {a},{b}")

    print("Heuristic schedule: all checks passed ✅")

# ===============================
#  MAIN
# ===============================

if __name__ == "__main__":
    verify_data()
    visualize_jobs_and_ops()
    visualize_precedence_matrix()

    print("\n=== HEURISTIC (GT + ATCS) RUN ===")
    S_h, C_h, am, al, Cw, Cf, T, T_max, C_max = heuristic_gt_atcs(k1=2.0)

    check_solution_heuristic(S_h, C_h, am, al)

    print("\n===== Objective (Heuristic) =====")
    print(f"T_max = {T_max:.2f}")
    print(f"C_max = {C_max:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld={Cw[j]:.2f}, C_final={Cf[j]:.2f}, "
              f"T_j={T[j]:.2f}, d_j={d_j[j]}")

    print("\n===== Operations =====")
    for i in I:
        print(f"Op {i}: S={S_h[i]:.2f}, C={C_h[i]:.2f}, machine={am[i]}, station={al[i]}")

    plot_gantt_by_machine(S_h, C_h, am, title="Machine-wise welding schedule (Heuristic)")
    plot_gantt_by_station(S_h, C_h, al, title="Station-wise welding schedule (Heuristic)")
    plt.show()
