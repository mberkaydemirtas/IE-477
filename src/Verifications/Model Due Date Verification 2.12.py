# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 19:31:52 2025

@author: Dell
"""
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from check_gurobi import check_solution_gurobi


# ===============================
#  DATA – SCENARIO: One Very Urgent Job (Job 1, 10 operations)
# ===============================

# Jobs
J = [1, 2, 3, 4, 5, 6, 7, 8]

# Operations
I = list(range(1, 31))  # 1..30

# Job -> operations mapping (O_j)
# Job 1: 10 operations (urgent job)
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

# ===============================
#  MACHINES, MACHINE TYPES & STATIONS
# ===============================

# 4 fiziksel makine
M = [1, 2, 3, 4]

# 2 makine tipi (örnek):
# 1 = TIG, 2 = MAG
K = [1, 2]

# Her makinenin tipi:
# Makine 1-2: TIG, 3-4: MAG (örnek)
machine_type = {
    1: 1,  # TIG
    2: 1,  # TIG
    3: 2,  # MAG
    4: 2,  # MAG
}

# Her operasyon için UYGUN makine tip(ler)i (K_i)
# ÖRNEK PATTERN (bazıları iki tipte de yapılabiliyor):
#  - i % 3 == 1 → sadece TIG (tip 1)
#  - i % 3 == 2 → sadece MAG (tip 2)
#  - i % 3 == 0 → hem TIG hem MAG (tip 1 ve 2)
K_i = {}
for i in I:
    r = i % 3
    if r == 1:
        K_i[i] = [1]        # sadece TIG
    elif r == 2:
        K_i[i] = [2]        # sadece MAG
    else:
        K_i[i] = [1, 2]     # her iki tipte de olabilir

# FEASIBLE MACHINE SETS M_i:
# M_i[i] = {m ∈ M : machine_type[m] ∈ K_i[i]}
M_i = {
    i: [m for m in M if machine_type[m] in K_i[i]]
    for i in I
}

# STATIONS
L = [1, 2, 3, 4]

# Şimdilik tüm operasyonlar tüm istasyonlara gidebiliyor
L_i = {i: [1, 2, 3, 4] for i in I}

# Big and small stations
L_big   = [1, 2, 3]
L_small = [4]

# ===============================
#  PRECEDENCE RELATIONS (Aynen bırakıyoruz)
# ===============================

# Başlangıçta hepsi boş
Pred_i = {i: [] for i in I}

# Her job içinde lineer sıra: op_k-1 -> op_k
for j in J:
    ops = O_j[j]
    for k in range(1, len(ops)):
        prev_op = ops[k - 1]
        curr_op = ops[k]
        Pred_i[curr_op].append(prev_op)

# EK: her job'un son operasyonu, o jobtaki bütün önceki operasyonlardan sonra gelsin
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for i_op in ops:
        if i_op != last:
            preds.add(i_op)
    Pred_i[last] = list(preds)

# ===============================
#  PROCESSING TIMES
# ===============================

p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)          # 3..7 range
        machine_add = (m - 1) * 0.5
        p_im[(i, m)] = float(base + machine_add)

# ===============================
#  RELEASE TIMES – hepsi 0 (sadece due date farkı çalışsın)
# ===============================

r_j = {j: 0.0 for j in J}

# ===============================
#  DUE DATES – Job 1 is extremely urgent
# ===============================

d_j = {
    1: 25.0,   # << ÇOK SIKI due date, 10 operasyonlu acil iş
    2: 80.0,
    3: 90.0,
    4: 85.0,
    5: 100.0,
    6: 95.0,
    7: 110.0,
    8: 120.0,
}

# ===============================
#  Grinding Requirement g_j
# ===============================

g_j = {
    1: 1,
    2: 0,
    3: 1,
    4: 0,
    5: 1,
    6: 0,
    7: 1,
    8: 0,
}

# ===============================
#  Painting Requirement p_j
# ===============================

p_j = {
    1: 1,
    2: 1,
    3: 1,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
}

# ===============================
#  Grinding & Painting Times
# ===============================

t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)     # 2 veya 3
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

# ===============================
#  BIG-STATION REQUIREMENT
# ===============================

Iend = [O_j[j][-1] for j in J]  # her job'un son operasyonu
beta_i = {i: (1 if i in Iend else 0) for i in I}

# ===============================
#  Big-M Values
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
visualize_precedence_matrix(I, Pred_i)

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
        print(f"Op {i}: S = {S[i].X:.2f}, C = {C[i].X:.2f}, "
              f"machine = {m_sel}, station = {l_sel}")

    # Gantt grafikleri
    plot_gantt_by_machine(I, M, M_i, x, S, C,
                          title="Machine-wise welding schedule")
    plot_gantt_by_station(I, L, L_i, y, S, C,
                          title="Station-wise welding schedule")

    plt.show()
