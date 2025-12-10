# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 19:53:03 2025

@author: Dell
"""

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Welding Shop Scheduling – LARGE EXAMPLE (8 jobs, 30 ops)
Verification + visualization:

- Number of operations per job
- Release and due dates per job
- GLOBAL precedence table + matrix (all operations)
- Machine-wise welding schedule
- Station-wise welding schedule

SCENARIO: Machine TYPE 2 is overloaded and has only 2 machines
"""

from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from matplotlib.patches import Patch   # <-- EKLENDİ: legend & bar chart için
from check_gurobi import check_solution_gurobi

# ===============================
#  DATA – SCENARIO: Many ops require TYPE 2, but only 2 type-2 machines
# ===============================

# Jobs
J = [1, 2, 3, 4]

# Operations (1..30)
I = list(range(1, 31))

# -------------------------------------------------
# JOB STRUCTURE
# -------------------------------------------------
O_j = {
    1: list(range(1, 16)),     # ops 1–15
    2: list(range(16, 26)),    # ops 16–25
    3: [26, 27, 28],           # small job
    4: [29, 30],               # small job
}

# ===============================
#  MACHINES & MACHINE TYPES
# ===============================

# 4 fiziksel makine
M = [1, 2, 3, 4]

# 2 makine tipi
# 1 = Type1, 2 = Type2 (overloaded type)
K_types = [1, 2]

# Her makinenin tipi:
# Makine 1–2: Type 2
# Makine 3–4: Type 1
machine_type = {
    1: 2,  # type 2
    2: 2,  # type 2
    3: 1,  # type 1
    4: 1,  # type 1
}

# Her operasyon için UYGUN makine tipleri K_i
# - Op 1–24: sadece type 2 (darboğaz)
# - Op 25–30: hem type 1 hem type 2
K_i = {}
for i in I:
    if i <= 24:
        K_i[i] = [2]          # çoğu iş type 2 istiyor
    else:
        K_i[i] = [1, 2]       # son 6 op esnek

# MACHINE FEASIBILITY M_i: tiplerden türetiliyor
# M_i[i] = { m ∈ M : machine_type[m] ∈ K_i[i] }
M_i = {
    i: [m for m in M if machine_type[m] in K_i[i]]
    for i in I
}

# ===============================
#  STATIONS
# ===============================

L = [1, 2, 3, 4]

# Stations (all allowed)
L_i = {i: [1, 2, 3, 4] for i in I}

# Big/small stations
L_big   = [1, 2, 3]
L_small = [4]

# ===============================
# PRECEDENCE — Each job simple chain
# ===============================

Pred_i = {i: [] for i in I}

for j in J:
    ops = O_j[j]
    # zincir: 1->2->3->...
    for k in range(1, len(ops)):
        Pred_i[ops[k]].append(ops[k-1])

    # Last operation depends on all previous ops in the job
    last = ops[-1]
    preds = set(Pred_i[last])
    for op in ops[:-1]:
        preds.add(op)
    Pred_i[last] = list(preds)

# ===============================
# PROCESSING TIMES
# ===============================

p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)
        machine_offset = (m - 1) * 0.3
        p_im[(i, m)] = float(base + machine_offset)

# ===============================
# RELEASE TIMES — all zero
# ===============================

r_j = {j: 0.0 for j in J}

# ===============================
# DUE DATES — moderate, not urgent
# ===============================

d_j = {
    1: 100.0,
    2: 110.0,
    3: 120.0,
    4: 130.0,
}

# ===============================
# Grinding Requirements
# ===============================

g_j = {1: 1, 2: 0, 3: 1, 4: 0}

# ===============================
# Painting Requirements
# ===============================

p_j = {1: 1, 2: 1, 3: 0, 4: 0}

# Grinding & Painting times
t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

# ===============================
# BIG-STATION NEEDS
# Only last operations need big station
# ===============================

Iend = [O_j[j][-1] for j in J]
beta_i = {i: (1 if i in Iend else 0) for i in I}

# ===============================
# Big-M Values
# ===============================

M_proc = 1000.0
M_seq  = 1000.0
M_Lseq = 1000.0


# ==========================================================
#  DATA VERIFICATION + VISUALIZATION FUNCTIONS
# ==========================================================

def verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j):
    print("\n=== DATA VERIFICATION ===")

    # 1) O_j içindeki tüm operasyonlar I'de mi?
    all_ops_from_jobs = set()
    for j in J:
        for i in O_j[j]:
            all_ops_from_jobs.add(i)
            if i not in I:
                print(f"[WARNING] Job {j} operation {i} not in I!")
    missing_ops = set(I) - all_ops_from_jobs
    if missing_ops:
        print(f"[WARNING] Some operations in I are not in any job O_j: {missing_ops}")
    else:
        print("OK: All operations in I are covered by some O_j.")

    # 2) Her op için feasible machine & station var mı?
    bad_m = [i for i in I if len(M_i[i]) == 0]
    bad_l = [i for i in I if len(L_i[i]) == 0]
    if bad_m:
        print(f"[ERROR] Some operations have no feasible machines: {bad_m}")
    else:
        print("OK: Every operation has at least one feasible machine.")
    if bad_l:
        print(f"[ERROR] Some operations have no feasible stations: {bad_l}")
    else:
        print("OK: Every operation has at least one feasible station.")

    # 3) Pred_i referansları valid mi?
    bad_pred = []
    for i in I:
        for h in Pred_i[i]:
            if h not in I:
                bad_pred.append((i, h))
    if bad_pred:
        print(f"[ERROR] Some Pred_i refer to non-existing operations: {bad_pred}")
    else:
        print("OK: All predecessors in Pred_i are valid operations.")

    # 4) p_im tanımlı mı?
    missing_p = []
    for i in I:
        for m in M_i[i]:
            if (i, m) not in p_im:
                missing_p.append((i, m))
    if missing_p:
        print(f"[ERROR] Missing p_im entries for: {missing_p}")
    else:
        print("OK: All (i,m) with m in M_i[i] have p_im defined.")

    # 5) r_j ve d_j tam mı?
    if set(r_j.keys()) != set(J):
        print("[ERROR] r_j is not defined for all jobs!")
    else:
        print("OK: r_j defined for all jobs.")
    if set(d_j.keys()) != set(J):
        print("[ERROR] d_j is not defined for all jobs!")
    else:
        print("OK: d_j defined for all jobs.")


def visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_j, t_grind_j, t_paint_j):
    """
    Çizilenler:
    - Number of operations per job
    - Release and due dates per job
    (Precedence görselleştirmesi ayrı fonksiyonda)
    """
    print("\n=== JOB – OP SUMMARY ===")
    for j in J:
        ops = O_j[j]
        print(f"Job {j}: ops = {ops}, "
              f"r_j = {r_j[j]}, d_j = {d_j[j]}, "
              f"grind = {g_j[j]}, paint = {p_j[j]}, "
              f"t_grind = {t_grind_j[j]}, t_paint = {t_paint_j[j]}")
    
    # 1) Number of operations per job
    job_ids = J
    op_counts = [len(O_j[j]) for j in J]
    
    plt.figure()
    plt.bar(job_ids, op_counts)
    plt.xlabel("Job")
    plt.ylabel("Number of operations")
    plt.title("Number of operations per job")
    plt.tight_layout()
    
    # 2) Release and due dates per job (eski hali)
    releases = [r_j[j] for j in J]
    dues     = [d_j[j] for j in J]
    
    plt.figure()
    plt.scatter(job_ids, releases, marker="o", label="release")
    plt.scatter(job_ids, dues,     marker="x", label="due")
    plt.xlabel("Job")
    plt.ylabel("Time")
    plt.title("Release and due dates per job")
    plt.legend()
    plt.tight_layout()


def visualize_job_operation_membership(J, O_j):
    """
    Hangi job hangi global operasyonlara sahip?
    Y ekseni: Job ID
    X ekseni: Global operasyon numarası (i)
    Her nokta: (i, j) yani 'job j, op i' bağlantısı
    """
    plt.figure(figsize=(10, 4))
    
    for j in J:
        ops = O_j[j]
        x_vals = ops
        y_vals = [j] * len(ops)
        plt.scatter(x_vals, y_vals, label=f"Job {j}")
        for i in ops:
            plt.text(i, j + 0.05, str(i), ha="center", va="bottom", fontsize=7)

    plt.yticks(J, [f"Job {j}" for j in J])
    plt.xlabel("Global operation index (i)")
    plt.ylabel("Job")
    plt.title("Job → Operation membership (which job owns which operations)")
    plt.grid(True, axis="x", linestyle=":", linewidth=0.5)
    plt.tight_layout()


def visualize_precedence_matrix(I, Pred_i):
    """
    GLOBAL precedence table + matrix:

    - Prints all precedence relations h -> i
    - Draws an |I| x |I| matrix: row = predecessor h, col = successor i
    """
    print("\n=== GLOBAL PRECEDENCE LIST (h -> i) ===")
    for i in I:
        for h in Pred_i[i]:
            print(f"{h} -> {i}")

    # build matrix
    n = len(I)
    index_of = {op: idx for idx, op in enumerate(I)}
    mat = [[0] * n for _ in range(n)]
    for i in I:
        ci = index_of[i]
        for h in Pred_i[i]:
            if h in index_of:
                rh = index_of[h]
                mat[rh][ci] = 1

    # plot matrix
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Precedence (1 = h -> i)")
    plt.xticks(range(n), I, rotation=90)
    plt.yticks(range(n), I)
    plt.xlabel("Successor operation i")
    plt.ylabel("Predecessor operation h")
    plt.title("Precedence matrix for all operations")
    plt.tight_layout()


def visualize_big_station_needs(J, O_j, beta_i):
    """
    Big station requirement tablosu / grafiği:

    - Y ekseni: Job'lar
    - X ekseni: Job içindeki operasyon pozisyonu (1., 2., 3. ... gibi)
    - Her operasyon için ayrı bar çiziliyor (aynı job satırı üzerinde yatay).
    - Her barın içine operasyon numarası yazılıyor.
    - beta_i = 1 (big zorunlu) ve beta_i = 0 (esnek) için farklı renkler.
    """
    print("\n=== BIG STATION REQUIREMENT SUMMARY ===")
    for j in J:
        ops = O_j[j]
        must_big = sum(1 for i in ops if beta_i[i] == 1)
        flex     = sum(1 for i in ops if beta_i[i] == 0)
        print(f"Job {j}: must_big = {must_big}, flexible = {flex}, total = {len(ops)}")

    plt.figure()
    ax = plt.gca()

    for j_idx, j in enumerate(J):
        ops = O_j[j]
        for pos_idx, i in enumerate(ops):
            left = pos_idx
            width = 0.9

            if beta_i[i] == 1:
                color = "tab:orange"  # Must be big
            else:
                color = "tab:blue"    # Flexible

            ax.barh(j_idx, width, left=left, color=color)
            ax.text(left + width / 2.0, j_idx, f"{i}",
                    va="center", ha="center", fontsize=8)

    ax.set_yticks(range(len(J)))
    ax.set_yticklabels(J)
    ax.set_xlabel("Operation position within job")
    ax.set_ylabel("Job")
    ax.set_title("Big-station requirement per job (operation-level)")

    legend_handles = [
        Patch(facecolor="tab:blue",  label="Flexible (β_i = 0)"),
        Patch(facecolor="tab:orange", label="Must be big (β_i = 1)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()


def plot_gantt_by_machine(I, M, M_i, x, S, C, title="Machine-wise welding schedule"):
    """
    Her makine için zaman ekseninde operasyon bloklarını çiz.
    GÜNCEL: Aynı job aynı renk, sağda legend paneli.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    y_ticks = []
    y_labels = []

    # Op -> job map
    op_to_job = {}
    for j in J:
        for op in O_j[j]:
            op_to_job[op] = j

    # Job colors
    cmap = plt.cm.get_cmap("tab10")
    job_colors = {j: cmap((j - 1) % 10) for j in J}

    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        mtype = machine_type[m]
        y_labels.append(f"Machine {m} (Type {mtype})")

        for i in I:
            if m not in M_i[i]:
                continue
            if x[i, m].X > 0.5:
                start = S[i].X
                finish = C[i].X
                width = finish - start

                job_of_i = op_to_job.get(i, None)
                color = job_colors.get(job_of_i, "gray")

                ax.barh(y_pos, width, left=start, color=color)
                ax.text(start + width / 2, y_pos, f"{i}",
                        va="center", ha="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    # Legend panel
    handles = [
        Patch(facecolor=job_colors[j], label=f"Job {j}")
        for j in J
    ]
    ax.legend(handles=handles, title="Jobs",
              loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])


def plot_gantt_by_station(I, L, L_i, y, S, C, title="Station-wise welding schedule"):
    """
    Her istasyon için zaman ekseninde operasyon bloklarını çiz.
    GÜNCEL: Aynı job aynı renk, sağda legend paneli.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    y_ticks = []
    y_labels = []

    # Op -> job map
    op_to_job = {}
    for j in J:
        for op in O_j[j]:
            op_to_job[op] = j

    cmap = plt.cm.get_cmap("tab10")
    job_colors = {j: cmap((j - 1) % 10) for j in J}

    for idx_l, l in enumerate(L):
        y_pos = idx_l
        # Big/small vurgusu istersen:
        if l in L_big:
            y_labels.append(f"Station {l} (big)")
        else:
            y_labels.append(f"Station {l} (small)")
        y_ticks.append(y_pos)

        for i in I:
            if l not in L_i[i]:
                continue
            if y[i, l].X > 0.5:
                start = S[i].X
                finish = C[i].X
                width = finish - start

                job_of_i = op_to_job.get(i, None)
                color = job_colors.get(job_of_i, "gray")

                ax.barh(y_pos, width, left=start, color=color)
                ax.text(start + width / 2, y_pos, f"{i}",
                        va="center", ha="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)

    handles = [
        Patch(facecolor=job_colors[j], label=f"Job {j}")
        for j in J
    ]
    ax.legend(handles=handles, title="Jobs",
              loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout(rect=[0, 0, 0.8, 1])


# ===============================
#  MODEL
# ===============================

model = Model("Welding_Full_LaTeX_Integrated_Type2Overload")

# x_{im} only for m in M_i[i]
x = model.addVars(
    [(i, m) for i in I for m in M_i[i]],
    vtype=GRB.BINARY,
    name="x"
)

# y_{iℓ} for ℓ in L_i[i]
y = model.addVars(
    [(i, l) for i in I for l in L_i[i]],
    vtype=GRB.BINARY,
    name="y"
)

# sequencing z_{ii'm}, z_{ii'ℓ}
zM_index = [(i, h, m) for i in I for h in I if i != h for m in M]
zM = model.addVars(zM_index, vtype=GRB.BINARY, name="zM")

zL_index = [(i, h, l) for i in I for h in I if i != h for l in L]
zL = model.addVars(zL_index, vtype=GRB.BINARY, name="zL")


# times
S = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="S")
C = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

C_weld  = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="C_weld")
C_final = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="C_final")

T     = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="T")
T_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T_max")
C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

# OBJECTIVE
model.setObjective(T_max, GRB.MINIMIZE)

# ===============================
#  CONSTRAINTS
# ===============================

# 1) Machine assignment
for i in I:
    model.addConstr(
        quicksum(x[i, m] for m in M_i[i]) == 1,
        name=f"assign_M_{i}"
    )

# 2) Station assignment
for i in I:
    model.addConstr(
        quicksum(y[i, l] for l in L_i[i]) == 1,
        name=f"assign_L_{i}"
    )

# 3) Processing time linking
for i in I:
    for m in M_i[i]:
        pijm = p_im[(i, m)]
        model.addConstr(
            C[i] - S[i] >= pijm - M_proc * (1 - x[i, m]),
            name=f"ptime_lb_{i}_{m}"
        )
        model.addConstr(
            C[i] - S[i] <= pijm + M_proc * (1 - x[i, m]),
            name=f"ptime_ub_{i}_{m}"
        )

# 4) Station size feasibility
for i in I:
    if beta_i[i] == 1:
        for l in L_small:
            if l in L_i[i]:
                model.addConstr(y[i, l] == 0, name=f"small_forbid_{i}_{l}")

# 5) Precedence
for i in I:
    for h in Pred_i[i]:
        model.addConstr(S[i] >= C[h], name=f"prec_{h}_{i}")

# 6) Earliest start
for j in J:
    first_op = O_j[j][0]
    model.addConstr(S[first_op] >= r_j[j], name=f"release_{j}")

# 7–8) Machine sequencing
I_list = I[:]
nI = len(I_list)

for idx_i in range(nI):
    for idx_h in range(idx_i + 1, nI):
        i = I_list[idx_i]
        h = I_list[idx_h]

        common_machines = set(M_i[i]).intersection(M_i[h])

        for m in common_machines:
            model.addConstr(
                S[h] >= C[i]
                       - M_seq * (1 - zM[i, h, m])
                       - M_seq * (1 - x[i, m])
                       - M_seq * (1 - x[h, m]),
                name=f"no_overlapM_h_after_i_{i}{h}{m}"
            )

            model.addConstr(
                S[i] >= C[h]
                       - M_seq * zM[i, h, m]
                       - M_seq * (1 - x[i, m])
                       - M_seq * (1 - x[h, m]),
                name=f"no_overlapM_i_after_h_{i}{h}{m}"
            )

            model.addConstr(
                zM[i, h, m] <= x[i, m],
                name=f"zM_le_xi_{i}{h}{m}"
            )
            model.addConstr(
                zM[i, h, m] <= x[h, m],
                name=f"zM_le_xh_{i}{h}{m}"
            )
            model.addConstr(
                zM[i, h, m] >= x[i, m] + x[h, m] - 1,
                name=f"zM_ge_sum_{i}{h}{m}"
            )

# 9–10) Station sequencing
for idx_i in range(nI):
    for idx_h in range(idx_i + 1, nI):
        i = I_list[idx_i]
        h = I_list[idx_h]

        common_stations = set(L_i[i]).intersection(L_i[h])

        for l in common_stations:
            model.addConstr(
                S[h] >= C[i]
                       - M_Lseq * (1 - zL[i, h, l])
                       - M_Lseq * (1 - y[i, l])
                       - M_Lseq * (1 - y[h, l]),
                name=f"no_overlapL_h_after_i_{i}{h}{l}"
            )
            model.addConstr(
                S[i] >= C[h]
                       - M_Lseq * zL[i, h, l]
                       - M_Lseq * (1 - y[i, l])
                       - M_Lseq * (1 - y[h, l]),
                name=f"no_overlapL_i_after_h_{i}{h}{l}"
            )

            model.addConstr(
                zL[i, h, l] <= y[i, l],
                name=f"zL_le_yi_{i}{h}{l}"
            )
            model.addConstr(
                zL[i, h, l] <= y[h, l],
                name=f"zL_le_yh_{i}{h}{l}"
            )
            model.addConstr(
                zL[i, h, l] >= y[i, l] + y[h, l] - 1,
                name=f"zL_ge_sum_{i}{h}{l}"
            )

# 11) Welding completion
for j in J:
    last_op = O_j[j][-1]
    model.addConstr(C_weld[j] == C[last_op], name=f"Cweld_{j}")

# 12) Final completion
for j in J:
    model.addConstr(
        C_final[j] == C_weld[j]
                     + g_j[j] * t_grind_j[j]
                     + p_j[j] * t_paint_j[j],
        name=f"Cfinal_{j}"
    )

# 13) Tardiness
for j in J:
    model.addConstr(T[j] >= C_final[j] - d_j[j], name=f"Tdef_{j}")

# 14) Makespan and maximum tardiness
for j in J:
    model.addConstr(C_max >= C_final[j], name=f"Cmax_ge_{j}")
    model.addConstr(T_max >= T[j],      name=f"Tmax_ge_{j}")

# ===============================
#  DATA VERIFICATION + VISUALIZATION (before solve)
# ===============================

verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j)
visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_j, t_grind_j, t_paint_j)
visualize_job_operation_membership(J, O_j)
visualize_precedence_matrix(I, Pred_i)
visualize_big_station_needs(J, O_j, beta_i)

# ===============================
#  SOLVE
# ===============================

model.optimize()

# ===============================
#  PRINT & GANTT
# ===============================

if model.SolCount == 0:
    print("\nNo feasible solution. Status =", model.Status)
else:
    check_solution_gurobi(
        J, I, O_j,
        M, L,
        M_i, L_i,
        Pred_i,
        r_j, d_j,
        g_j, p_j,
        t_grind_j, t_paint_j,
        x, y, S, C,
        C_weld, C_final,
        T, T_max, C_max,
        tol=1e-6
    )
    print("\n===== Objective =====")
    print(f"T_max = {T_max.X:.2f}")
    print(f"C_max = {C_max.X:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld = {C_weld[j].X:.2f}, "
              f"C_final = {C_final[j].X:.2f}, "
              f"T_j = {T[j].X:.2f}, d_j = {d_j[j]}")

    print("\n===== Operations =====")
    for i in I:
        m_list = [m for m in M_i[i] if x[i, m].X > 0.5]
        l_list = [l for l in L_i[i] if y[i, l].X > 0.5]
        m_sel = m_list[0] if m_list else None
        l_sel = l_list[0] if l_list else None
        if m_sel is not None:
            mtype = machine_type[m_sel]
        else:
            mtype = "-"
        print(f"Op {i}: S = {S[i].X:.2f}, C = {C[i].X:.2f}, "
              f"machine = {m_sel} (Type {mtype}), station = {l_sel}")

    # Gantt grafikleri
    plot_gantt_by_machine(I, M, M_i, x, S, C,
                          title="Machine-wise welding schedule (with types)")
    plot_gantt_by_station(I, L, L_i, y, S, C,
                          title="Station-wise welding schedule")

    plt.show()
