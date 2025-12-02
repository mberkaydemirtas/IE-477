#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welding Shop Scheduling – LARGE EXAMPLE (8 jobs, 30 ops)
Verification + visualization:

- Number of operations per job
- Release and due dates per job
- GLOBAL precedence table + matrix (all operations)
- Job-based precedence list
- Job-based small precedence graphs
- Machine-wise welding schedule
- Station-wise welding schedule
"""

from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

# ===============================
#  DATA (LARGE EXAMPLE – 8 jobs, 30 ops)
# ===============================

# Jobs
J = [1, 2, 3, 4, 5, 6, 7, 8]

# Operations
I = list(range(1, 31))  # 1..30

# Job -> operations mapping (O_j)
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

# Machines and stations
M = [1, 2, 3, 4]
L = [1, 2, 3, 4]

# Feasible machine sets M_i
M_i = {}
for i in I:
    mod = i % 4
    if mod == 1:          # pattern 1
        M_i[i] = [1, 2]
    elif mod == 2:        # pattern 2
        M_i[i] = [2, 3]
    elif mod == 3:        # pattern 3
        M_i[i] = [2, 3, 4]
    else:                 # mod == 0, pattern 4
        M_i[i] = [1, 3, 4]

# Feasible station sets L_i (şimdilik hepsi tüm istasyonlara gidebilir)
L_i = {i: [1, 2, 3, 4] for i in I}

# Big and small stations
L_big   = [1, 2, 3]
L_small = [4]

# ===============================
#  PRECEDENCE (bazıları bağlı, bazıları bağımsız)
# ===============================

Pred_i = {i: [] for i in I}

# Job 1: ops = [1, 2, 3]
# 1 → 2, 1 → 3
Pred_i[2] = [1]
Pred_i[3] = [1]

# Job 2: ops = [4, 5, 6, 7]
# 4 → 5, 4 → 6, 5 → 7
Pred_i[5] = [4]
Pred_i[6] = [4]
Pred_i[7] = [5]

# Job 3: ops = [8, 9, 10, 11, 12]
# 8 → 9, 8 → 10, 9 & 10 → 11, 11 → 12
Pred_i[9]  = [8]
Pred_i[10] = [8]
Pred_i[11] = [9, 10]
Pred_i[12] = [11]

# Job 4: ops = [13, 14]
# 13 → 14
Pred_i[14] = [13]

# Job 5: ops = [15, 16, 17, 18]
# 15 → 16, 15 → 17, 16 → 18
Pred_i[16] = [15]
Pred_i[17] = [15]
Pred_i[18] = [16]

# Job 6: ops = [19, 20, 21]
# 19 & 20 → 21
Pred_i[21] = [19, 20]

# Job 7: ops = [22, 23, 24, 25, 26]
# 22 → 23, 22 → 24, 23 → 25, 24 → 26
Pred_i[23] = [22]
Pred_i[24] = [22]
Pred_i[25] = [23]
Pred_i[26] = [24]

# Job 8: ops = [27, 28, 29, 30]
# İç bağlantı yok (şimdilik)

# EK ADIM: Her job'un son operasyonu, o jobtaki tüm operasyonlardan sonra gelsin
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for i_op in ops:
        if i_op == last:
            continue
        preds.add(i_op)
    Pred_i[last] = list(preds)

# Processing times p_im
p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)          # 3..7 arası base
        machine_add = (m - 1) * 0.5 # makine indexine göre fark
        p_im[(i, m)] = float(base + machine_add)

# Release times r_j (0,2,4,...,14)
r_j = {}
release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
for idx, j in enumerate(J):
    r_j[j] = release_times[idx]

# Due dates d_j
d_i = {}
due_base = 30.0
Iend = [O_j[j][-1] for j in J]  # last op of each job
for idx, i_last in enumerate(Iend):
    d_i[i_last] = due_base + 3.0 * idx   # 30,33,36,...,51

d_j = {}
for j in J:
    last_op = O_j[j][-1]
    d_j[j] = d_i[last_op]

# Grinding requirement g_j
g_j = {}
for idx, j in enumerate(J):
    if idx % 2 == 0:   # j=1,3,5,7
        g_j[j] = 1
    else:              # j=2,4,6,8
        g_j[j] = 0

# Painting requirement p_j
p_j = {}
for idx, j in enumerate(J):
    if idx in [0, 1, 2]:  # j=1,2,3
        p_j[j] = 1
    else:
        p_j[j] = 0

# Grinding and painting times
t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    if p_j[j] == 1:
        t_paint_j[j] = 3.0
    else:
        t_paint_j[j] = 0.0

# Big-station requirement beta_i (son operasyonlar zorunlu, diğerleri 0)
beta_i = {}
for i in I:
    if i in Iend:
        beta_i[i] = 1
    else:
        beta_i[i] = 0

# Big-M values
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
    
    # 2) Release and due dates per job
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


def print_precedence_by_job(J, O_j, Pred_i):
    """
    Her job için: her operasyonun predecessor listesini yazdır.
    """
    print("\n=== PRECEDENCE BY JOB ===")
    for j in J:
        print(f"\nJob {j}: ops = {O_j[j]}")
        for i in O_j[j]:
            preds = Pred_i[i]
            print(f"  Op {i}: preds = {preds}")


def visualize_precedence_per_job(J, O_j, Pred_i):
    """
    Her job için küçük bir precedence grafiği:
    - X ekseni: job içindeki sıralama
    - Üzerine oklarla h -> i bağları
    """
    for j in J:
        ops = O_j[j]
        plt.figure()
        x_pos = {i: k for k, i in enumerate(ops)}  # job içindeki index
        y = 0.0

        # düğümler
        for i in ops:
            x = x_pos[i]
            plt.scatter(x, y)
            plt.text(x, y + 0.05, str(i), ha="center")

        # oklar (h -> i) – sadece aynı job içindeki predecessor'ları çiz
        for i in ops:
            for h in Pred_i[i]:
                if h not in ops:
                    continue
                x1, y1 = x_pos[h], y
                x2, y2 = x_pos[i], y
                plt.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->")
                )

        plt.title(f"Job {j} precedence graph")
        plt.yticks([])
        plt.xticks(list(x_pos.values()), [str(i) for i in ops])
        plt.xlabel("Operation order in job")
        plt.tight_layout()


def visualize_precedence_matrix_with_job_blocks(I, Pred_i, O_j, J):
    """
    GLOBAL precedence matrix:
    - matrix[h, i] = 1 if h -> i
    - job blok sınırlarını çiz
    """
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
            if h in index_of:
                rh = index_of[h]
                mat[rh][ci] = 1

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Precedence (1 = h -> i)")
    plt.xticks(range(n), I, rotation=90)
    plt.yticks(range(n), I)
    plt.xlabel("Successor operation i")
    plt.ylabel("Predecessor operation h")
    plt.title("Precedence matrix with job blocks")

    # job blok sınır çizgileri
    pos = 0
    for j in J:
        size = len(O_j[j])
        plt.axvline(pos - 0.5, linestyle="--")
        plt.axhline(pos - 0.5, linestyle="--")
        pos += size
    plt.axvline(pos - 0.5, linestyle="--")
    plt.axhline(pos - 0.5, linestyle="--")

    plt.tight_layout()


def plot_gantt_by_machine(I, M, M_i, x, S, C, title="Machine-wise welding schedule"):
    """
    Her makine için zaman ekseninde operasyon bloklarını çiz.
    """
    plt.figure()
    y_ticks = []
    y_labels = []

    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        y_labels.append(f"Machine {m}")

        for i in I:
            if m not in M_i[i]:
                continue
            if x[i, m].X > 0.5:
                start = S[i].X
                finish = C[i].X
                plt.barh(y_pos, finish - start, left=start)
                plt.text(start, y_pos, f"{i}", va="center")

    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


def plot_gantt_by_station(I, L, L_i, y, S, C, title="Station-wise welding schedule"):
    """
    Her istasyon için zaman ekseninde operasyon bloklarını çiz.
    """
    plt.figure()
    y_ticks = []
    y_labels = []

    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)
        y_labels.append(f"Station {l}")

        for i in I:
            if l not in L_i[i]:
                continue
            if y[i, l].X > 0.5:
                start = S[i].X
                finish = C[i].X
                plt.barh(y_pos, finish - start, left=start)
                plt.text(start, y_pos, f"{i}", va="center")

    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()

# ===============================
#  MODEL
# ===============================

model = Model("Welding_Full_LaTeX_Integrated")

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
zM = model.addVars(I, I, M, vtype=GRB.BINARY, name="zM")
zL = model.addVars(I, I, L, vtype=GRB.BINARY, name="zL")

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

# precedence’i daha okunur göster:
print_precedence_by_job(J, O_j, Pred_i)
visualize_precedence_per_job(J, O_j, Pred_i)
visualize_precedence_matrix_with_job_blocks(I, Pred_i, O_j, J)

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
        print(f"Op {i}: S = {S[i].X:.2f}, C = {C[i].X:.2f}, "
              f"machine = {m_sel}, station = {l_sel}")

    # Gantt grafikleri
    plot_gantt_by_machine(I, M, M_i, x, S, C,
                          title="Machine-wise welding schedule")
    plot_gantt_by_station(I, L, L_i, y, S, C,
                          title="Station-wise welding schedule")

    plt.show()
